# Renode-Modeled Accelerator Evaluation

This repository contains a small cat/dog classifier for an RV32IM firmware target plus a Renode peripheral that models activation and convolution offload. The accelerator numbers in this repository are now explicitly **Renode-modeled simulated results**, not measured FPGA timing.

## What Changed

- The Renode peripheral no longer returns convolution work with effectively zero simulated cost.
- Accelerator commands now schedule completion through Renode `LocalTimeSource` virtual time using:
  - convolution compute
  - startup overhead
  - memory transfer
  - batch activation
- Evaluation images are exported as an external Renode memory blob instead of being embedded as a large firmware array.
- Benchmark outputs are written to `results/` with accuracy and Wilson confidence intervals.
- The external evaluation blob is now `renode/test_images.bin`, loaded into SRAM at `0x00100000`.

## Reproducible Commands

Export a Renode evaluation subset:

```bash
./venv/bin/python scripts/export_eval_dataset.py --count 500
```

Run one reproduced sweep point:

```bash
./venv/bin/python scripts/benchmark_renode.py --dataset-count 100 --macs-per-cycle 4
```

Run one large full-UART benchmark point and preserve the entire UART log:

```bash
./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 500 --macs-per-cycle 4 --timeout 7200
```

In this environment, a single monolithic 500-image Renode run is not stable enough to treat as canonical. The reproducible packaging flow is therefore chunked:

```bash
./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 200 --dataset-offset 0   --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0000_count_0200_mpc_4.txt
./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 200 --dataset-offset 200 --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0200_count_0200_mpc_4.txt
./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 100 --dataset-offset 400 --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0400_count_0100_mpc_4.txt
./venv/bin/python scripts/build_chunked_submission_artifacts.py
```

This preserves three real UART logs plus explicit offset metadata and builds a single canonical `results/benchmark_500.csv` from them. The packaged benchmark artifacts are generated from the chunk manifest recorded in `results/benchmark_500_manifest.json`, not from ad hoc manual headless logs. The canonical packaging flow is:

- `renode/run_headless_autostop.resc`
- `scripts/benchmark_renode.py`
- `scripts/run_renode_full_uart.py`
- `results/chunked/*.txt` plus adjacent `*.json`
- `scripts/build_chunked_submission_artifacts.py`
- `results/benchmark_500.csv`
- `results/benchmark_500_manifest.json`

The combined 500-image benchmark produces:

- `results/benchmark_500.csv`
- `results/table1_accuracy_500.csv`
- `results/float_reference_accuracy.csv`

Then verify the UART/CSV pair:

```bash
./venv/bin/python scripts/verify_benchmark_consistency.py
```

`scripts/build_500_sweep_from_benchmark.py` remains a derived sensitivity script. It is not a true rerun script and should not be treated as canonical reproduced benchmark evidence.

## Cycle Model

The Renode peripheral in `renode/ecp5_platform.repl` executes functionally in host-side Python and releases `ACC_STATUS` / `ACC_CONV_STATUS` via `machine.LocalTimeSource.ExecuteInSyncedState(...)`. The scheduled delay is computed from:

- Convolution:
  - exact same-padding MAC count
  - `RENODE_MACS_PER_CYCLE`
  - `RENODE_CONV_STARTUP_CYCLES`
  - transfer cost from input, weights, bias, and output bytes
- Batch activation:
  - `RENODE_ACTIVATION_ELEMS_PER_CYCLE`
  - `RENODE_BATCH_STARTUP_CYCLES`
  - transfer cost from source and destination bytes

Current defaults are controlled by environment variables read by the Renode peripheral:

- `RENODE_MACS_PER_CYCLE` default `4`
- `RENODE_ACTIVATION_ELEMS_PER_CYCLE` default `8 * MACS_PER_CYCLE`
- `RENODE_BYTES_PER_CYCLE` default `8`
- `RENODE_CONV_STARTUP_CYCLES` default `40`
- `RENODE_BATCH_STARTUP_CYCLES` default `12`

The `ACC_MODEL_CYCLES` register remains as a diagnostic counter, but the benchmarked accelerator totals now come from the CLINT timer advancing in Renode virtual time while firmware waits on the status bit. These are author-defined timing assumptions for architecture exploration. They are not cycle-accurate RTL measurements.

## Results

Generated artifacts live in `results/`:

- `chunked/renode_direct_uart_offset_*.txt`: canonical preserved chunk UART logs for `benchmark_500.csv`
- `benchmark_500.csv`: exact 500-row benchmark export aggregated from the canonical chunk UART logs
- `benchmark_500_manifest.json`: source-log manifest for `benchmark_500.csv`
- `table1_accuracy_500.csv`: 500-image accuracy table data with CI rows and float-vs-Q8.8 delta row
- `float_reference_accuracy.csv`: float-model accuracy on the same evaluation subset
- `eval_dataset_manifest.csv`: exported evaluation subset manifest

Generated figures live in `figs/`:

- `speedup_sweep.pdf`: only present when a true rerun sweep is generated separately

The authoritative source for `benchmark_500.csv` is the chunk list recorded in `benchmark_500_manifest.json`.

See `FIX_REPORT.md` for the full honesty review and remaining limitations.
