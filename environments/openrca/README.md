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

1. Download the OpenRCA telemetry dataset from [Google Drive](https://drive.google.com/drive/folders/1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM)
2. Place the data so the directory structure is:
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
| `data_dir` | `"dataset"` | Path to the OpenRCA dataset directory |
| `systems` | All systems | List of systems to include: `"Bank"`, `"Market/cloudbed-1"`, `"Market/cloudbed-2"`, `"Telecom"` |
| `max_turns` | `30` | Maximum tool-use turns per rollout |
| `num_examples` | `-1` | Number of examples (-1 for all) |

## How It Works

The environment provides two tools for the model to analyze telemetry data:

1. **`execute_python`** — Runs Python code in a persistent environment with pandas, numpy, and other libraries pre-imported. The `DATA_DIR` variable points to the telemetry data root.

2. **`list_directory`** — Lists files and directories within the telemetry data directory for exploration.

The model iteratively analyzes metrics, traces, and logs to identify root causes, then provides a structured JSON answer with the root cause datetime, component, and/or reason.

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
