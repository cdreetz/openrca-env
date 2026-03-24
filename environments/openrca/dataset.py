"""
Dataset builder for the OpenRCA benchmark.

Loads query CSV files from the OpenRCA dataset and converts them into
a HuggingFace Dataset suitable for the Verifiers framework.
"""

import json
import os

import pandas as pd
from datasets import Dataset

from .prompts import SYSTEM_INFO


def build_dataset(data_dir: str, systems: list[str]) -> Dataset:
    """Build a Verifiers Dataset from OpenRCA query CSV files.

    Each row becomes a rollout with a prompt containing the system-specific
    telemetry schema, root cause candidates, and the task instruction.

    Args:
        data_dir: Path to the OpenRCA dataset root directory.
        systems: List of system names to include.

    Returns:
        A HuggingFace Dataset with prompt, answer, and info columns.

    Raises:
        FileNotFoundError: If query CSV files are not found.
    """
    rows = []
    for system in systems:
        query_path = os.path.join(data_dir, system, "query.csv")
        if not os.path.exists(query_path):
            raise FileNotFoundError(
                f"Query file not found at {query_path}. "
                f"Please download the OpenRCA dataset from "
                f"https://drive.google.com/drive/folders/"
                f"1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM "
                f"and place it in '{data_dir}/'."
            )

        df = pd.read_csv(query_path)
        info = SYSTEM_INFO[system]

        for _, row in df.iterrows():
            instruction = row["instruction"]
            task_index = row["task_index"]
            scoring_points = row["scoring_points"]

            task_id = int(task_index.split("_")[1])
            if task_id <= 3:
                difficulty = "easy"
            elif task_id <= 6:
                difficulty = "medium"
            else:
                difficulty = "hard"

            user_content = (
                f"{info['schema']}\n\n"
                f"{info['candidates']}\n\n"
                f"## TASK\n\n"
                f"{instruction}\n\n"
                f"Use the `execute_python` tool to write and run Python code "
                f"for analyzing the telemetry data. Use the `list_directory` "
                f"tool to explore available data files. The `DATA_DIR` variable "
                f"in the Python environment points to the telemetry data root."
            )

            rows.append(
                {
                    "prompt": [{"role": "user", "content": user_content}],
                    "answer": scoring_points,
                    "info": json.dumps(
                        {
                            "system": system,
                            "task_index": task_index,
                            "difficulty": difficulty,
                        }
                    ),
                }
            )

    return Dataset.from_list(rows)
