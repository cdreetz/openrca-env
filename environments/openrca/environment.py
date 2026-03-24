"""
OpenRCA SandboxEnv implementation.

Extends PythonEnv to provide per-rollout sandbox isolation with a persistent
Python REPL and file exploration tools for analyzing telemetry data.
"""

import asyncio
import json
import os
import shlex
import tarfile
import tempfile
from typing import Any

import verifiers as vf
from verifiers.envs.python_env import PythonEnv
from verifiers.envs.sandbox_env import SandboxState


class OpenRCAEnv(PythonEnv):
    """OpenRCA environment for root cause analysis of software failures.

    Each rollout gets its own isolated sandbox container with a persistent
    Python REPL and the relevant system's telemetry data uploaded into it.
    This ensures that even if the agent writes or deletes files, it cannot
    affect other concurrent rollouts.

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
        memory_gb: int = 2,
        disk_size_gb: int = 10,
        timeout_per_command_seconds: int = 60,
        **kwargs: Any,
    ) -> None:
        self.local_data_dir = os.path.abspath(data_dir)
        self.sandbox_data_dir = sandbox_data_dir
        super().__init__(
            pip_install_packages="pandas numpy pytz",
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_per_command_seconds=timeout_per_command_seconds,
            **kwargs,
        )
        self.add_tool(
            self.list_directory,
            args_to_skip=["sandbox_id", "sandbox_state"],
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create sandbox, upload system data, and initialize Python REPL."""
        state = await super().setup_state(state, **kwargs)

        sandbox_id: str = state["sandbox_id"]
        sandbox_state: SandboxState = state["sandbox_state"]

        # Determine which system's data to upload from the rollout info
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        system: str = info.get("system", "")

        if system:
            await self._upload_system_data(sandbox_id, sandbox_state, system)

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

    async def _upload_system_data(
        self,
        sandbox_id: str,
        sandbox_state: SandboxState,
        system: str,
    ) -> None:
        """Upload system telemetry data to the sandbox as a tar archive.

        If the local data directory is not found, a warning is logged and the
        upload is skipped — the sandbox is expected to have data available via
        a custom Docker image or volume mount instead.
        """
        local_system_dir = os.path.join(self.local_data_dir, system)
        if not os.path.isdir(local_system_dir):
            self.logger.warning(
                f"Local data directory not found: {local_system_dir}. "
                f"Skipping data upload — ensure data is available in the "
                f"sandbox via a custom Docker image or volume mount."
            )
            return

        # Wait for sandbox to be ready before uploading
        if not sandbox_state["ready"]:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)

        # Create tar archive of system data and upload to sandbox
        fd, tmp_tar = tempfile.mkstemp(suffix=".tar.gz")
        os.close(fd)
        try:
            await asyncio.to_thread(
                self._create_tar, tmp_tar, local_system_dir, system
            )

            remote_tar = "/tmp/openrca_data.tar.gz"
            await self.sandbox_client.upload_file(
                sandbox_id, remote_tar, tmp_tar
            )
            await self.bash(
                f"mkdir -p {shlex.quote(self.sandbox_data_dir)} && "
                f"tar xzf {remote_tar} -C {shlex.quote(self.sandbox_data_dir)} && "
                f"rm -f {remote_tar}",
                sandbox_id,
                sandbox_state,
            )
        finally:
            await asyncio.to_thread(self._cleanup_tar, tmp_tar)

    @staticmethod
    def _create_tar(tmp_tar: str, local_dir: str, arcname: str) -> None:
        """Create a gzipped tar archive (runs in a thread)."""
        with tarfile.open(tmp_tar, "w:gz") as tar:
            tar.add(local_dir, arcname=arcname)

    @staticmethod
    def _cleanup_tar(tmp_tar: str) -> None:
        """Remove temporary tar file if it exists (runs in a thread)."""
        if os.path.exists(tmp_tar):
            os.unlink(tmp_tar)

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
        # Validate path stays within the data directory
        full_path = os.path.normpath(
            os.path.join(self.sandbox_data_dir, path)
        )
        if full_path != self.sandbox_data_dir and not full_path.startswith(self.sandbox_data_dir + os.sep):
            return "Error: Access denied. Path must be within the data directory."

        # Run ls inside the sandbox
        return await self.bash(
            f"ls -la {shlex.quote(full_path)} 2>&1",
            sandbox_id,
            sandbox_state,
        )
