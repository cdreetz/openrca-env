# OpenRCA Environment

A Verifiers environment implementing the [OpenRCA benchmark](https://github.com/microsoft/OpenRCA) (ICLR 2025) for evaluating and training LLMs on root cause analysis of software failures.

## Overview

OpenRCA assesses LLMs' ability to identify root causes of failures in enterprise software systems by analyzing telemetry data (metrics, traces, and logs). The benchmark includes 335 failures across three systems:

- **Bank** — A banking microservice platform
- **Market** — An E-commerce platform with two cloudbeds
- **Telecom** — A telecom database system

Tasks range from easy (identify one root cause element) to hard (identify all three: datetime, component, and reason).

## Setup

### Data

The telemetry dataset (~65 GB) is hosted on [HuggingFace](https://huggingface.co/datasets/cdreetz/OpenRCA) and downloaded automatically — no manual setup needed. Each sandbox downloads the relevant system's data on rollout start.

The host also auto-downloads `query.csv` files on first run for dataset building. To pre-download:

```bash
python -c "from src.download import download_dataset; download_dataset('dataset')"
```

### Installation

```bash
prime env install openrca-env
```

## Usage

### Evaluation

```bash
# Quick smoke test (1 system, few examples)
vf-eval openrca_env -m gpt-4.1-mini -n 5 -a '{"systems": ["Telecom"]}'

# Full evaluation on a specific system
vf-eval openrca_env -m gpt-4.1-mini -n 50 -r 1 -s -a '{"systems": ["Bank"]}'

# All systems
vf-eval openrca_env -m gpt-4.1-mini -n 20 -r 1 -s
```

### Configuration

Pass these via `-a` (env args):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `"dataset"` | Path to query.csv files (auto-downloaded) |
| `systems` | All systems | `"Bank"`, `"Market/cloudbed-1"`, `"Market/cloudbed-2"`, `"Telecom"` |
| `max_turns` | `30` | Maximum tool-use turns per rollout |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `docker_image` | `"python:3.11-slim"` | Docker image for sandbox containers |
| `sandbox_data_dir` | `"/data/openrca"` | Path inside sandbox where telemetry data is placed |

## How It Works

Each rollout runs in its own **isolated sandbox container** (via `PythonEnv` / `SandboxEnv`).

Per-rollout flow:
1. A sandbox container is created (4 GB RAM, 50 GB disk)
2. `huggingface_hub` is installed and the relevant system's telemetry data is downloaded from [cdreetz/OpenRCA](https://huggingface.co/datasets/cdreetz/OpenRCA)
3. A persistent Python REPL is initialized with pandas, numpy, pytz, and `DATA_DIR`

The environment exposes two tools:

1. **`python`** — Persistent Python REPL with common imports pre-loaded. `DATA_DIR` points to the telemetry data root.

2. **`list_directory`** — Lists files and directories within the telemetry data directory.

The model iteratively analyzes metrics, traces, and logs to identify root causes, then provides a structured JSON answer.

### Scoring

Predictions are scored using the original OpenRCA evaluation methodology:
- **Component match**: Exact string match against ground truth
- **Reason match**: Exact string match against ground truth
- **Time match**: Within 1 minute of ground truth timestamp
- For multiple root causes, all permutations are evaluated to find the best matching

### Metrics

- `openrca_score`: Primary reward (0.0-1.0) based on scoring criteria match
- `difficulty_metric`: Task difficulty level (0=easy, 1=medium, 2=hard)

## Project Structure

```
openrca_env/
  openrca_env.py       # load_environment() + OpenRCAEnv class
  src/
    dataset.py          # Dataset builder (query.csv -> HF Dataset)
    download.py         # Auto-download from HuggingFace
    evaluation.py       # Scoring logic (ported from OpenRCA)
    prompts.py          # System prompts, telemetry schemas, candidates
  pyproject.toml
  README.md
```

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
