# Changes Audit

## Scope

This file records the repository changes made to:

- align software and accelerator math
- repair the Renode benchmark flow
- replace unstable canonical benchmark packaging with a reproducible chunked flow
- remove stale/conflicting result artifacts
- add explicit verification for numerical consistency and benchmark consistency

The accelerator in this repository remains a **Renode Python MMIO model**, not RTL or FPGA hardware.

## 1. Numerical Alignment Fixes

### 1.1 Sigmoid mismatch fixed

Previous state:

- Software sigmoid in `firmware/src/nn_ops.c` used a piecewise-linear Q8.8 approximation:
  - `x <= -1024 -> 0`
  - `x >= 1024 -> 256`
  - else `x / 8 + 128`, then clamp to `[0, 256]`
- Renode accelerator sigmoid in `renode/ecp5_platform.repl` used logistic `exp()` with rounding.

Change:

- The Renode accelerator sigmoid was changed to match the software sigmoid exactly.

Why:

- This was the first confirmed numerical divergence between the software path and the accelerator path.
- The benchmark is only fair if software and accelerator implement the same math.

Files changed:

- `renode/ecp5_platform.repl`

### 1.2 Accumulator-width mismatch fixed

Previous state:

- Software convolution and FC used `int32_t` accumulators.
- Renode convolution accumulation used unbounded Python integers.

Change:

- The Renode convolution path now uses signed-32-bit wrap helpers before the final Q8.8 rescale and int16 clamp.
- Helper functions added:
  - `to_s32()`
  - `add_s32()`
  - `mul_s32()`

Why:

- The Renode model must follow the software path’s arithmetic semantics as closely as possible.

Files changed:

- `renode/ecp5_platform.repl`

## 2. Renode Launch and UART Flow Fixes

### 2.1 Repository-local Renode path handling fixed

Previous state:

- The checked-in `.resc` flow depended on fragile path behavior and had previously pointed at another repository copy on disk.

Change:

- Renode benchmark launches now run from `renode/`.
- `.resc` files use repository-relative assets.

Files changed:

- `renode/run_headless.resc`
- `renode/run_headless_autostop.resc`
- `scripts/benchmark_renode.py`

### 2.2 UART backend path stabilized

Previous state:

- Renode failed to create the checked-in relative UART backend path reliably.

Change:

- Automated benchmark UART output now uses:
  - `/tmp/ai_classification_renode_uart_output.txt`
- Manual headless UART output now uses:
  - `/tmp/ai_classification_renode_manual_uart.txt`

Files changed:

- `renode/run_headless.resc`
- `renode/run_headless_autostop.resc`
- `scripts/benchmark_renode.py`
- `scripts/run_renode_full_uart.py`

### 2.3 Autostop control fixed

Previous state:

- The long benchmark flow exited early because the Renode control script was not robust for long runs.
- One attempted `sleep` value overflowed Renode’s internal millisecond conversion.

Change:

- `run_headless_autostop.resc` now:
  - attaches a UART line hook on `"Done."`
  - uses a valid long `sleep 7200`

Files changed:

- `renode/run_headless_autostop.resc`

### 2.4 Renode console failure capture added

Previous state:

- Early benchmark failures surfaced only as generic Python wrapper errors.

Change:

- `scripts/benchmark_renode.py` now stores Renode console output in:
  - `/tmp/ai_classification_renode_console.txt`
- On early exit, the script includes the console tail in the raised error.

Files changed:

- `scripts/benchmark_renode.py`

## 3. Numerical Verification Added

### 3.1 Layer-by-layer comparison script added

New script:

- `scripts/compare_numeric_paths.py`

What it does:

- parses exported firmware weights
- loads the local CIFAR-10 cat/dog evaluation subset
- evaluates:
  - software reference math
  - legacy accelerator math
  - current accelerator math
- reports:
  - first layer where divergence appears
  - per-layer mismatch counts
  - per-layer max absolute error
  - prediction mismatches
  - accumulator magnitude
  - observed signed-32-bit wrap events

Observed result on a 100-image run:

- Before fix:
  - first divergence: `sigmoid`
  - no earlier layer mismatches
- After fix:
  - no mismatches across checked layers
  - no prediction mismatches

## 4. Chunked Benchmark Redesign

### 4.1 Why the canonical benchmark is now chunked

Observed issue:

- A monolithic 500-image Renode run at `mpc=4` was not stable in this environment.
- Renode was killed by the OS during long runs.
- This made a single preserved 500-image UART file non-reproducible as a canonical artifact.

Evidence:

- Renode console tail captured:
  - `/usr/bin/renode: line 11: ... Killed`

Decision:

- The canonical `benchmark_500.csv` is now built from **three real preserved chunk UART runs** instead of one unstable monolithic run.

### 4.2 Dataset chunking support added

Change:

- `scripts/export_eval_dataset.py` now supports:
  - `--offset`

Files changed:

- `scripts/export_eval_dataset.py`

### 4.3 Benchmark driver offset support added

Change:

- `scripts/benchmark_renode.py` now supports:
  - `--dataset-offset`

Files changed:

- `scripts/benchmark_renode.py`

### 4.4 Full-UART preservation now stores chunk metadata

Change:

- `scripts/run_renode_full_uart.py` now supports:
  - `--dataset-offset`
- It writes metadata JSON next to each preserved UART log.
- It creates the destination directory before copying.

Files changed:

- `scripts/run_renode_full_uart.py`

### 4.5 Canonical chunked packaging script added

New script:

- `scripts/build_chunked_submission_artifacts.py`

What it does:

- reads preserved chunk UART logs plus metadata JSON
- validates each chunk
- combines them into:
  - `results/benchmark_500.csv`
  - `results/benchmark_500_manifest.json`
  - `results/table1_accuracy_500.csv`
  - `results/float_reference_accuracy.csv`

### 4.6 Consistency verifier updated for chunk manifests

Change:

- `scripts/verify_benchmark_consistency.py` now supports:
  - a manifest with `source_uart_logs`
- It aggregates rows across chunk logs and verifies exact agreement with `benchmark_500.csv`

Files changed:

- `scripts/verify_benchmark_consistency.py`

## 5. Latency / FPS Reporting Added

New script:

- `scripts/build_latency_fps_summary.py`

Generated outputs:

- `results/benchmark_500_latency_fps.csv`
- `results/latency_fps_summary.csv`

What it reports:

- per-image latency and FPS for accelerator and software
- average latency and FPS for accelerator and software

## 6. Documentation Updates

Files updated:

- `README.md`
- `results/BENCHMARK_ARTIFACTS.md`
- `scripts/build_submission_artifacts.py`
- `scripts/build_500_sweep_from_benchmark.py`

Main documentation changes:

- chunked UART logs are now documented as the canonical source for `benchmark_500.csv`
- the old single-log canonical flow is no longer the documented default
- the derived sweep helper is explicitly labeled as **derived** and **not a true rerun**

## 7. Result Artifact Cleanup

Removed stale/conflicting files:

- `results/renode_direct_uart.txt.1`
- `results/renode_direct_uart.txt.2`
- `results/renode_direct_uart.txt.3`
- `results/renode_direct_uart.txt.4`
- `results/renode_direct_uart.txt.5`
- `results/renode_direct_uart_authoritative.txt`
- `results/renode_direct_uart_stale_mismatch.txt`
- `results/intermediate/`
- `results/renode_runs_mpc_1.csv`
- `results/renode_runs_mpc_2.csv`
- `results/renode_runs_mpc_4.csv`
- `results/renode_runs_mpc_8.csv`
- `results/renode_runs_mpc_16.csv`
- `results/renode_summary_mpc_1.csv`
- `results/renode_summary_mpc_2.csv`
- `results/renode_summary_mpc_4.csv`
- `results/renode_summary_mpc_8.csv`
- `results/renode_summary_mpc_16.csv`
- `results/renode_sweep.csv`
- `results/speedup_sweep.csv`
- `results/renode_speedup_vs_mpc.png`
- `figs/speedup_sweep.pdf`

Why removed:

- They were either:
  - legacy conflicting UART logs
  - partial or stale results
  - derived sweep artifacts no longer corresponding to the current canonical benchmark source

## 8. Current Canonical Benchmark Files

Canonical preserved UART logs:

- `results/chunked/renode_direct_uart_offset_0000_count_0200_mpc_4.txt`
- `results/chunked/renode_direct_uart_offset_0200_count_0200_mpc_4.txt`
- `results/chunked/renode_direct_uart_offset_0400_count_0100_mpc_4.txt`

Canonical packaged outputs:

- `results/benchmark_500.csv`
- `results/benchmark_500_manifest.json`
- `results/table1_accuracy_500.csv`
- `results/float_reference_accuracy.csv`
- `results/benchmark_500_latency_fps.csv`
- `results/latency_fps_summary.csv`

## 9. Verified Current Results

From `results/benchmark_500.csv` and `results/benchmark_500_manifest.json`:

- row count: `500`
- average accelerator cycles: `244316.202`
- average software cycles: `23130259.156`
- average speedup: `94.673456`
- accelerator accuracy: `73.600%`
- software accuracy: `73.600%`
- prediction mismatches between accelerator and software: `0`

Modeled accelerator cycles at `mpc=4`:

- `156838`

Latency / FPS summary from `results/latency_fps_summary.csv`:

- Accelerator:
  - average latency: `9.772648 ms`
  - average FPS: `102.326411`
- Software:
  - average latency: `925.210366 ms`
  - average FPS: `1.080835`

Consistency verification:

- `scripts/verify_benchmark_consistency.py` reports `PASS`

## 10. Remaining Limitations

- The accelerator is still a Renode Python model, not real RTL.
- A true multi-point rerun sweep for `macs_per_cycle = {1, 2, 4, 8, 16}` was not rebuilt in chunked form here.
- The canonical benchmark is now trustworthy for the chunked `mpc=4` flow, not for an absent true-rerun sweep.
- Full 500-image layer-by-layer numeric verification was not completed; direct layer-by-layer proof was completed on a 100-image sample, while the 500-row benchmark itself was verified through exact UART/CSV consistency checks.
