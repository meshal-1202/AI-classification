# Renode Accelerator Honesty Fix Report

## What Was Wrong

- The Renode convolution path in [renode/ecp5_platform.repl](/home/pc/Desktop/AI classification/renode/ecp5_platform.repl:13) executed the full convolution inside host-side Python and returned completion immediately.
- Firmware timing used the CLINT cycle counter alone, so the host-side convolution looked almost free in simulated time.
- The old benchmark parser only reported the first matching result instead of aggregating the full run set.
- The old firmware benchmark repeated a tiny 10-image subset and embedded images directly in firmware headers.
- Paper and documentation text still described these Renode numbers as measured, which overstated hardware realism.

## Files Changed

- [renode/ecp5_platform.repl](/home/pc/Desktop/AI classification/renode/ecp5_platform.repl:13)
- [renode/run_headless.resc](/home/pc/Desktop/AI classification/renode/run_headless.resc:1)
- [renode/run_headless_autostop.resc](/home/pc/Desktop/AI classification/renode/run_headless_autostop.resc:1)
- [firmware/include/platform.h](/home/pc/Desktop/AI classification/firmware/include/platform.h:5)
- [firmware/src/main.c](/home/pc/Desktop/AI classification/firmware/src/main.c:1)
- [scripts/export_eval_dataset.py](/home/pc/Desktop/AI classification/scripts/export_eval_dataset.py:1)
- [scripts/benchmark_renode.py](/home/pc/Desktop/AI classification/scripts/benchmark_renode.py:1)
- [scripts/aggregate_renode_sweep.py](/home/pc/Desktop/AI classification/scripts/aggregate_renode_sweep.py:1)
- [scripts/build_submission_artifacts.py](/home/pc/Desktop/AI classification/scripts/build_submission_artifacts.py:1)
- [train_catdog.py](/home/pc/Desktop/AI classification/train_catdog.py:1)
- [README.md](/home/pc/Desktop/AI classification/README.md:1)
- [paper_latex/main.tex](/home/pc/Desktop/AI classification/paper_latex/main.tex:1)
- [paper_latex/README.md](/home/pc/Desktop/AI classification/paper_latex/README.md:1)
- [ECP5_ACCELERATION_CHANGES.md](/home/pc/Desktop/AI classification/ECP5_ACCELERATION_CHANGES.md:1)
- Generated artifacts in [results](/home/pc/Desktop/AI classification/results)
- Generated dataset metadata in [firmware/include/eval_dataset_meta.h](/home/pc/Desktop/AI classification/firmware/include/eval_dataset_meta.h:1)

## Code And Workflow Changes

- Switched the Renode peripheral to Renode `LocalTimeSource` virtual-time scheduling so accelerator completion is released at a future virtual timestamp instead of returning immediately from the Python callback.
- Added an explicit Renode cycle model for convolution:
  - exact same-padding MAC count
  - configurable `RENODE_MACS_PER_CYCLE`
  - configurable startup overhead
  - configurable transfer cost from input, weight, bias, and output bytes
- Added an explicit Renode cycle model for batch activation:
  - configurable element throughput
  - configurable startup overhead
  - configurable transfer cost
- Kept the modeled-cycle MMIO register only as a diagnostic counter; the reported benchmark totals now come from the CLINT timer alone.
- Switched benchmark wording in firmware output from generic latency/FPS to simulated latency/FPS.
- Moved Renode evaluation images out of firmware rodata and into an external blob named `test_images.bin` loaded into SRAM at `0x00100000`.
- Added reproducible export of evaluation labels and dataset manifest.
- Reworked the Renode benchmark script to:
  - rebuild firmware
  - export the evaluation subset
  - run Renode
  - parse all runs, not just the first
  - compute accuracy and Wilson 95% confidence intervals
  - compute float reference accuracy on the same subset
  - emit per-image CSVs, per-point summary CSVs, an aggregated sweep CSV, and a plot
- Added a validation check that fails if accelerated convolution stages regress back toward implausibly tiny values.

## Verified

- Python syntax checks:
  - `./venv/bin/python -m py_compile scripts/export_eval_dataset.py scripts/benchmark_renode.py scripts/aggregate_renode_sweep.py`
- Firmware build:
  - `make -C firmware`
- Renode smoke test:
  - `./venv/bin/python scripts/benchmark_renode.py --dataset-count 5 --macs-per-cycle 4 --timeout 600`
- Reproduced the canonical chunked 500-image benchmark packaging flow:
  - preserved chunk UART logs under `results/chunked/renode_direct_uart_offset_*_count_*_mpc_4.txt`
  - `./venv/bin/python scripts/build_chunked_submission_artifacts.py`
- Legacy-only single-UART packaging path:
  - `scripts/build_submission_artifacts.py` is kept only for intentional single-log rebuilds and is no longer the canonical packaged benchmark path
- Generated the full 500-image sweep from the completed 500-image benchmark trace and the exact Renode timing model:
  - `./venv/bin/python scripts/build_500_sweep_from_benchmark.py`

## Reproduced Outputs

- Sweep CSV: [results/renode_sweep.csv](/home/pc/Desktop/AI classification/results/renode_sweep.csv:1)
- PDF-requested sweep CSV: [results/speedup_sweep.csv](/home/pc/Desktop/AI classification/results/speedup_sweep.csv:1)
- Float reference CSV: [results/float_reference_accuracy.csv](/home/pc/Desktop/AI classification/results/float_reference_accuracy.csv:1)
- Exact 500-row benchmark CSV: [results/benchmark_500.csv](/home/pc/Desktop/AI classification/results/benchmark_500.csv:1)
- Table 1 accuracy CSV with CI and delta row: [results/table1_accuracy_500.csv](/home/pc/Desktop/AI classification/results/table1_accuracy_500.csv:1)
- Per-point summaries:
  - [results/renode_summary_mpc_1.csv](/home/pc/Desktop/AI classification/results/renode_summary_mpc_1.csv:1)
  - [results/renode_summary_mpc_2.csv](/home/pc/Desktop/AI classification/results/renode_summary_mpc_2.csv:1)
  - [results/renode_summary_mpc_4.csv](/home/pc/Desktop/AI classification/results/renode_summary_mpc_4.csv:1)
  - [results/renode_summary_mpc_8.csv](/home/pc/Desktop/AI classification/results/renode_summary_mpc_8.csv:1)
  - [results/renode_summary_mpc_16.csv](/home/pc/Desktop/AI classification/results/renode_summary_mpc_16.csv:1)
- Detailed default-point log CSV: [results/renode_runs_mpc_4.csv](/home/pc/Desktop/AI classification/results/renode_runs_mpc_4.csv:1)
- Plot: [results/renode_speedup_vs_mpc.png](/home/pc/Desktop/AI classification/results/renode_speedup_vs_mpc.png)
- PDF-requested figure: [figs/speedup_sweep.pdf](/home/pc/Desktop/AI classification/figs/speedup_sweep.pdf)
- Dataset manifest: [results/eval_dataset_manifest.csv](/home/pc/Desktop/AI classification/results/eval_dataset_manifest.csv:1)

## Key Reproduced Numbers

- Reproduced direct benchmark size: 500 CIFAR-10 cat/dog test images
- Quantized firmware accuracy on the 500-image benchmark: 73.6%
- Quantized Wilson 95% CI on the 500-image benchmark: 69.567% to 77.273%
- Float reference accuracy on the same 500-image subset: 73.6%
- Float-vs-Q8.8 delta on the 500-image subset: 0.0 percentage points
- Simulated software-only average on the completed 500-image direct benchmark: 23,149,998 cycles
- Simulated accelerator-assisted average on the completed 500-image direct benchmark at `MACS_PER_CYCLE=4`: 269,855 cycles
- Simulated speedup on the completed 500-image direct benchmark at `MACS_PER_CYCLE=4`: 85.79x
- Simulated software-only latency: about 926 ms per image at 25 MHz
- Simulated accelerator-assisted speedup range across the 500-image sweep: 32.49x to 145.41x

## Diff-Style Summary

- `renode/ecp5_platform.repl`: added modeled cycle accounting, `LocalTimeSource` scheduling, and SRAM-backed external dataset memory mapping.
- `firmware/include/platform.h`: added modeled-cycle MMIO accessors and reused SRAM for external dataset addressing.
- `firmware/src/main.c`: switched benchmark timing to CLINT-only virtual-time measurement, external dataset addressing, and simulated wording.
- `scripts/export_eval_dataset.py`: new external evaluation blob export path.
- `scripts/benchmark_renode.py`: rewritten to run reproducible Renode-modeled benchmarks with accuracy and CI reporting.
- `scripts/aggregate_renode_sweep.py`: new sweep aggregation helper.
- `scripts/build_submission_artifacts.py`: legacy-only single-UART packaging path for intentional rebuilds; canonical packaged `benchmark_500.csv` now comes from `scripts/build_chunked_submission_artifacts.py` plus `benchmark_500_manifest.json`.
- `scripts/build_500_sweep_from_benchmark.py`: generates the full 500-image sweep from the completed benchmark trace plus the exact Renode timing model.
- `train_catdog.py`: removed the old 10-image export from the main training flow.
- `README.md` and `paper_latex/*`: replaced overclaiming wording with simulated / Renode-modeled language and updated the reproduced numbers.

## Remaining Limitations

- The 500-image sweep artifacts are generated from the completed 500-image benchmark trace plus the exact Renode timing model, rather than from five separate completed full-Renode reruns. This is exact with respect to the repository timing model because `MACS_PER_CYCLE` changes only the modeled accelerator delay term, not the software baseline or predictions.
- The accelerator timing model is still an author-defined cost function, not cycle-accurate RTL timing.
- No FPGA bitstream, post-synthesis timing, area, or power measurements were generated in this fix pass.
- No local LaTeX engine (`pdflatex`, `lualatex`, `xelatex`, `tectonic`, or `latexmk`) was available here, so the updated manuscript source was not compiled locally.

## Honesty Review

- [ECP5_ACCELERATION_CHANGES.md](/home/pc/Desktop/AI classification/ECP5_ACCELERATION_CHANGES.md:1) was reduced to a short archival notice and no longer preserves the obsolete pre-fix performance claims.
- [train_catdog.py](/home/pc/Desktop/AI classification/train_catdog.py:89) still contains the historical helper that writes `test_images.h`. The main training flow no longer uses it, and the comment now says the current Renode evaluation path uses the external blob exporter.
- Any future claim that these results represent FPGA timing, cycle-accurate RTL timing, or measured hardware throughput would still be inaccurate unless new hardware evidence is added.
