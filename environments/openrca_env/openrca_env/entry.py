"""
OpenRCA Verifiers Environment — entry point.

Implements the OpenRCA benchmark (ICLR 2025) as a Verifiers environment for
evaluating and training LLMs on root cause analysis of software failures.

Reference: https://github.com/microsoft/OpenRCA
Paper: https://openreview.net/forum?id=M4qNIzQYpd
"""

import verifiers as vf

from .dataset import build_dataset
from .download import ensure_dataset
from .environment import OpenRCAEnv
from .evaluation import difficulty_metric, openrca_score
from .prompts import ALL_SYSTEMS, SYSTEM_INFO, SYSTEM_PROMPT


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
        data_dir: Path to the OpenRCA dataset directory on the host machine,
            containing system subdirectories (Bank/, Market/, Telecom/) with
            query.csv and telemetry data. Used both for building the task
            dataset and for uploading telemetry data into per-rollout sandboxes.
        systems: List of systems to include. Options: "Bank",
            "Market/cloudbed-1", "Market/cloudbed-2", "Telecom".
            Defaults to all systems.
        max_turns: Maximum number of tool-use turns per rollout.
        num_examples: Number of examples to use. -1 for all.
        docker_image: Docker image for sandbox containers. Use a custom image
            with pre-loaded data to avoid per-rollout uploads for large
            datasets.
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

    # Auto-download dataset from Google Drive if not present locally
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
