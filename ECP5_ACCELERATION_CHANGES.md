# Historical Archive Notice

This file previously contained pre-fix acceleration notes and obsolete performance claims.
Those claims are no longer part of the current evaluation and should not be cited.

Use these current sources instead:

- `FIX_REPORT.md`
- `README.md`
- `results/benchmark_500.csv`
- `results/speedup_sweep.csv`
- `results/table1_accuracy_500.csv`
- `paper_latex/main.tex`

Current status summary:

- accelerator timing is charged through Renode `LocalTimeSource`
- reported performance is simulated / Renode-modeled, not measured FPGA timing
- the benchmark uses an external dataset blob loaded by Renode
- current paper-facing artifacts are generated from the reproducible scripts in `scripts/`
