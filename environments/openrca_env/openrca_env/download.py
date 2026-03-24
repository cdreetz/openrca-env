"""
Automatic dataset downloader for the OpenRCA benchmark.

Downloads the telemetry dataset from HuggingFace when the local data
directory is missing or incomplete.
"""

import logging
import os

logger = logging.getLogger(__name__)

HF_REPO_ID = "cdreetz/OpenRCA"

EXPECTED_SYSTEMS = ["Bank", "Market", "Telecom"]


def is_dataset_present(data_dir: str, systems: list[str] | None = None) -> bool:
    """Check whether the dataset directory has the expected structure."""
    if not os.path.isdir(data_dir):
        return False

    if systems is None:
        systems = EXPECTED_SYSTEMS

    for system in systems:
        system_dir = os.path.join(data_dir, system)
        if not os.path.isdir(system_dir):
            return False
        query_path = os.path.join(system_dir, "query.csv")
        if not os.path.exists(query_path):
            has_cloudbed = any(
                os.path.exists(os.path.join(system_dir, cb, "query.csv"))
                for cb in os.listdir(system_dir)
                if os.path.isdir(os.path.join(system_dir, cb))
            )
            if not has_cloudbed:
                return False
    return True


def download_dataset(data_dir: str) -> None:
    """Download the OpenRCA dataset from HuggingFace.

    Args:
        data_dir: Destination directory for the dataset.

    Raises:
        ImportError: If huggingface_hub is not installed.
        RuntimeError: If the download fails or dataset is incomplete.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download the OpenRCA dataset. "
            "Install it with: pip install huggingface_hub"
        )

    os.makedirs(data_dir, exist_ok=True)

    logger.info(
        f"Downloading OpenRCA dataset from HuggingFace ({HF_REPO_ID}) "
        f"to {data_dir}..."
    )
    logger.info("This is ~65 GB of telemetry data and may take a while.")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=data_dir,
    )

    if not is_dataset_present(data_dir):
        raise RuntimeError(
            f"Download completed but dataset at {data_dir} appears "
            f"incomplete. Expected directories: {EXPECTED_SYSTEMS}."
        )

    logger.info(f"OpenRCA dataset ready at {data_dir}")


def ensure_dataset(data_dir: str) -> str:
    """Ensure the dataset is available, downloading if necessary.

    Args:
        data_dir: Path to the dataset root directory.

    Returns:
        The absolute path to the dataset directory.
    """
    data_dir = os.path.abspath(data_dir)

    if is_dataset_present(data_dir):
        logger.debug(f"Dataset already present at {data_dir}")
        return data_dir

    logger.info(
        f"Dataset not found at {data_dir}. "
        f"Downloading from HuggingFace..."
    )
    download_dataset(data_dir)
    return data_dir
