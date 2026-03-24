"""
OpenRCA Verifiers Environment.

Implements the OpenRCA benchmark (ICLR 2025) as a Verifiers environment for
evaluating and training LLMs on root cause analysis of software failures.

Reference: https://github.com/microsoft/OpenRCA
Paper: https://openreview.net/forum?id=M4qNIzQYpd
"""

import json
import os
import re
import shlex
from typing import Any

import verifiers as vf
from verifiers.envs.python_env import PythonEnv
from verifiers.envs.sandbox_env import SandboxState

from src.dataset import build_dataset
from src.download import ensure_dataset
from src.evaluation import difficulty_metric, openrca_score
from src.prompts import ALL_SYSTEMS, SYSTEM_INFO, SYSTEM_PROMPT

# HuggingFace dataset repo for the OpenRCA telemetry data
HF_REPO_ID = "cdreetz/OpenRCA"


# ─── Environment ─────────────────────────────────────────────────────────


class OpenRCAEnv(PythonEnv):
    """OpenRCA environment for root cause analysis of software failures.

    Each rollout gets its own isolated sandbox container with a persistent
    Python REPL and the relevant system's telemetry data downloaded into it
    from HuggingFace.

    Tools exposed to the model:
        - ``python``: Persistent Python REPL (inherited from PythonEnv)
        - ``list_directory``: Browse files within the telemetry data directory
    """

    def __init__(
        self,
        data_dir: str,
        sandbox_data_dir: str = "/data/openrca",
        docker_image: str = "python:3.11-slim",
        cpu_cores: int = 1,
        memory_gb: int = 4,
        disk_size_gb: int = 50,
        timeout_per_command_seconds: int = 300,
        **kwargs: Any,
    ) -> None:
        self.local_data_dir = os.path.abspath(data_dir)
        self.sandbox_data_dir = sandbox_data_dir
        # PythonEnv.__init__ hardcodes docker_image in its super().__init__
        # call alongside **kwargs, so passing SandboxEnv params both as
        # named args and in kwargs causes "multiple values" errors. Strip
        # them from kwargs, let PythonEnv use its defaults, then patch the
        # sandbox_request object with our actual values.
        for key in ("docker_image", "cpu_cores", "memory_gb",
                    "disk_size_gb", "timeout_per_command_seconds"):
            kwargs.pop(key, None)
        super().__init__(
            pip_install_packages="pandas numpy pytz",
            **kwargs,
        )
        self.sandbox_request.docker_image = docker_image
        self.sandbox_request.cpu_cores = cpu_cores
        self.sandbox_request.memory_gb = memory_gb
        self.sandbox_request.disk_size_gb = disk_size_gb
        self.timeout_per_command_seconds = timeout_per_command_seconds
        self.add_tool(
            self.list_directory,
            args_to_skip=["sandbox_id", "sandbox_state"],
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create sandbox, download system data, and initialize Python REPL."""
        state = await super().setup_state(state, **kwargs)

        sandbox_id: str = state["sandbox_id"]
        sandbox_state: SandboxState = state["sandbox_state"]

        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        system: str = info.get("system", "")

        if system:
            await self._download_system_data(sandbox_id, sandbox_state, system)

        # Set DATA_DIR in the persistent Python REPL
        setup_code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import os, json, re\n"
            "from datetime import datetime, timedelta\n"
            "import pytz\n"
            "pd.set_option('display.width', 427)\n"
            "pd.set_option('display.max_columns', 10)\n"
            "pd.set_option('display.max_rows', 50)\n"
            f"DATA_DIR = {self.sandbox_data_dir!r}\n"
        )
        await self.python(
            setup_code,
            sandbox_id,
            sandbox_state,
            state["python_state"],
        )

        return state

    async def _download_system_data(
        self,
        sandbox_id: str,
        sandbox_state: SandboxState,
        system: str,
    ) -> None:
        """Download system telemetry data from HuggingFace into the sandbox."""
        if not sandbox_state["ready"]:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)

        top_system = system.split("/")[0]

        self.logger.info(
            f"Downloading {top_system} data from HuggingFace "
            f"into sandbox {sandbox_id}"
        )

        data_dir = self.sandbox_data_dir
        pattern = f"{top_system}/**"

        # Write the download script as a standalone Python file to avoid
        # shell quoting issues with nested python -c
        py_script = (
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download(\n"
            f"    repo_id={HF_REPO_ID!r},\n"
            f"    repo_type='dataset',\n"
            f"    allow_patterns={pattern!r},\n"
            f"    local_dir={data_dir!r},\n"
            f")\n"
        )

        download_script = (
            f"set -e\n"
            f"pip install -q huggingface_hub\n"
            f"cat > /tmp/_hf_download.py << 'PYEOF'\n"
            f"{py_script}"
            f"PYEOF\n"
            f"python3 /tmp/_hf_download.py\n"
            f"rm -f /tmp/_hf_download.py\n"
            f"if [ ! -d {shlex.quote(data_dir)}/{shlex.quote(top_system)}/telemetry ]; then\n"
            f"    echo 'ERROR: telemetry dir not found after download'\n"
            f"    ls -la {shlex.quote(data_dir)}/ 2>&1\n"
            f"    exit 1\n"
            f"fi\n"
            f"echo \"SUCCESS: $(find {shlex.quote(data_dir)}/{shlex.quote(top_system)} -type f | wc -l) files\"\n"
        )
        result = await self.bash(download_script, sandbox_id, sandbox_state)

        if "ERROR:" in str(result):
            self.logger.error(
                f"Download failed in sandbox {sandbox_id}: {result}"
            )
        else:
            last_line = result.strip().splitlines()[-1] if result.strip() else "done"
            self.logger.info(
                f"Downloaded {top_system} in sandbox {sandbox_id}: {last_line}"
            )

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject sandbox state into tool calls."""
        updated_args = super().update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )
        if tool_name == "list_directory":
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
        return updated_args

    async def list_directory(
        self,
        path: str,
        sandbox_id: str,
        sandbox_state: SandboxState,
    ) -> str:
        """List files and directories within the telemetry data directory.

        Args:
            path: Relative path from the data root directory
                  (e.g., "Bank/telemetry/2021_03_05/metric").

        Returns:
            A listing of files and directories at the specified path.
        """
        full_path = os.path.normpath(
            os.path.join(self.sandbox_data_dir, path)
        )
        if full_path != self.sandbox_data_dir and not full_path.startswith(self.sandbox_data_dir + os.sep):
            return "Error: Access denied. Path must be within the data directory."

        return await self.bash(
            f"ls -la {shlex.quote(full_path)} 2>&1",
            sandbox_id,
            sandbox_state,
        )


# ─── Entry point ─────────────────────────────────────────────────────────


def load_environment(
    data_dir: str = "dataset",
    systems: list[str] | None = None,
    max_turns: int = 30,
    num_examples: int = -1,
    docker_image: str = "python:3.11-slim",
    sandbox_data_dir: str = "/data/openrca",
) -> vf.Environment:
    """Load the OpenRCA root cause analysis environment.

    Args:
        data_dir: Path to the dataset directory containing query.csv files.
            Auto-downloaded from HuggingFace if not present.
        systems: List of systems to include. Options: "Bank",
            "Market/cloudbed-1", "Market/cloudbed-2", "Telecom".
            Defaults to all systems.
        max_turns: Maximum number of tool-use turns per rollout.
        num_examples: Number of examples to use. -1 for all.
        docker_image: Docker image for sandbox containers.
        sandbox_data_dir: Path inside the sandbox where telemetry data is
            placed. The agent accesses data at this path via DATA_DIR.

    Returns:
        A Verifiers Environment ready for evaluation or training.
    """
    if systems is None:
        systems = list(ALL_SYSTEMS)

    for system in systems:
        if system not in SYSTEM_INFO:
            raise ValueError(
                f"Unknown system '{system}'. "
                f"Valid systems: {list(SYSTEM_INFO.keys())}"
            )

    # Auto-download dataset from HuggingFace if not present locally
    data_dir = ensure_dataset(data_dir)

    dataset = build_dataset(data_dir, systems)

    if 0 < num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))

    rubric = vf.Rubric(funcs=[openrca_score])
    rubric.add_metric(difficulty_metric)

    env = OpenRCAEnv(
        data_dir=data_dir,
        sandbox_data_dir=sandbox_data_dir,
        docker_image=docker_image,
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )

    return env
