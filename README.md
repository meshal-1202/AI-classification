# AI Classification with Renode Accelerator Model

This repository contains a small cat/dog image classifier together with an RV32IM firmware target and a Renode-based accelerator model for convolution and activation offload.

The project is intended for simulation and architecture exploration. Performance numbers in this repository are Renode-modeled simulated results, not measured FPGA hardware results.

## Project Structure

- `train_catdog.py`: trains the cat/dog classifier.
- `run_benchmark.sh`: starts the benchmark workflow.
- `firmware/`: embedded inference code and support files for the RV32IM target.
- `renode/`: Renode platform description and run scripts.
- `scripts/`: dataset export, benchmarking, and verification helpers.
- `results/`: benchmark outputs, manifests, and generated reports.
- `best_catdog.pth`: trained model checkpoint.

## Main Features

- Cat/dog classifier training in Python.
- Quantized inference flow for embedded firmware.
- Renode peripheral model for accelerator-assisted convolution and activation.
- Benchmark scripts for comparing software-only and accelerator-assisted execution.
- Exported dataset blobs for Renode-based evaluation.

## Quick Start

Create and activate a Python environment, then install the dependencies you need for training and scripts.

Train the model:

```bash
python train_catdog.py
```

Export an evaluation dataset:

```bash
python scripts/export_eval_dataset.py --count 500
```

Run the benchmark:

```bash
./run_benchmark.sh
```

Verify generated benchmark artifacts:

```bash
python scripts/verify_benchmark_consistency.py
```

## Notes

- `venv/` and local dataset files are excluded from git.
- The included benchmark files under `results/` are generated artifacts from previous runs.
- The repository currently keeps the trained checkpoint and selected benchmark outputs because they are part of the project deliverables.
