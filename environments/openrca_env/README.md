# OpenRCA Environment

A Verifiers environment implementing the [OpenRCA benchmark](https://github.com/microsoft/OpenRCA) (ICLR 2025) for evaluating and training LLMs on root cause analysis of software failures.

## Overview

OpenRCA assesses LLMs' ability to identify root causes of failures in enterprise software systems by analyzing telemetry data (metrics, traces, and logs). The benchmark includes 335 failures across three systems:

- **Bank** — A banking microservice platform
- **Market** — An E-commerce platform with two cloudbeds
- **Telecom** — A telecom database system

Tasks range from easy (identify one root cause element) to hard (identify all three: datetime, component, and reason).

## Setup

### Prerequisites

The dataset (~68 GB) is **downloaded automatically** from Google Drive on first run. Just install and go — if the `dataset/` directory is missing, it will be fetched via `gdown`.

To download manually or ahead of time:

```python
from openrca import download_dataset
download_dataset("dataset")
```

Or from the command line:

```bash
python -c "from openrca import download_dataset; download_dataset('dataset')"
```

<details>
<summary>Expected directory structure after download</summary>

```
dataset/
├── Bank/
│   ├── query.csv
│   ├── record.csv
│   └── telemetry/
├── Market/
│   ├── cloudbed-1/
│   │   ├── query.csv
│   │   ├── record.csv
│   │   └── telemetry/
│   └── cloudbed-2/
│       ├── query.csv
│       ├── record.csv
│       └── telemetry/
└── Telecom/
    ├── query.csv
    ├── record.csv
    └── telemetry/
```

</details>

### Installation

```bash
prime env install openrca
```

## Usage

### Evaluation

```bash
# Quick smoke test
prime eval run openrca -m gpt-4.1-mini -n 5

# Full evaluation on a specific system
prime eval run openrca -m gpt-4.1-mini -n 50 -r 1 -s -x '{"systems": ["Bank"]}'

# All systems
prime eval run openrca -m gpt-4.1-mini -n 20 -r 1 -s
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `"dataset"` | Path to the OpenRCA dataset directory on the host |
| `systems` | All systems | List of systems to include: `"Bank"`, `"Market/cloudbed-1"`, `"Market/cloudbed-2"`, `"Telecom"` |
| `max_turns` | `30` | Maximum tool-use turns per rollout |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `docker_image` | `"python:3.11-slim"` | Docker image for sandbox containers |
| `sandbox_data_dir` | `"/data/openrca"` | Path inside sandbox where telemetry data is placed |

## How It Works

Each rollout runs in its own **isolated sandbox container** (via `PythonEnv` → `SandboxEnv`). This ensures that even if the agent writes or deletes files via the Python REPL, it cannot affect other concurrent rollouts.

Per-rollout flow:
1. A sandbox container is created via the Prime Sandboxes API
2. The relevant system's telemetry data is tar'd, uploaded, and extracted into the sandbox
3. A persistent Python REPL worker is initialized with common imports and `DATA_DIR`

The environment exposes two tools:

1. **`python`** — Executes code in a persistent Python REPL inside the sandbox, with pandas, numpy, and other libraries pre-imported. The `DATA_DIR` variable points to the telemetry data root.

2. **`list_directory`** — Lists files and directories within the telemetry data directory inside the sandbox.

The model iteratively analyzes metrics, traces, and logs to identify root causes, then provides a structured JSON answer with the root cause datetime, component, and/or reason.

For large datasets, build a custom Docker image with data pre-loaded and pass it via `docker_image` to avoid per-rollout uploads.

### Scoring

Predictions are scored using the original OpenRCA evaluation methodology:
- **Component match**: Exact string match against ground truth
- **Reason match**: Exact string match against ground truth
- **Time match**: Within 1 minute of ground truth timestamp
- For multiple root causes, all permutations are evaluated to find the best matching

### Metrics

- `openrca_score`: Primary reward (0.0–1.0) based on scoring criteria match
- `difficulty_metric`: Task difficulty level (0=easy, 1=medium, 2=hard)

## System Requirements

- **Storage**: ~80 GB for the full telemetry dataset
- **Memory**: 32 GB recommended (telemetry files can be large)

## Reference

```bibtex
@inproceedings{xu2025openrca,
  title={OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?},
  author={Xu, Junjielong and Zhang, Qinan and Zhong, Zhiqing and He, Shilin and Zhang, Chaoyun and Lin, Qingwei and Pei, Dan and He, Pinjia and Zhang, Dongmei and Zhang, Qi},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=M4qNIzQYpd}
}
```
