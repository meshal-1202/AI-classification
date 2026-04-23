#!/usr/bin/env bash
set -euo pipefail

cd "/home/pc/Desktop/AI ./AI classification"

rm -f \
  results/uart_full_offset_0_count_200_mpc_4.txt \
  results/uart_full_offset_0_count_200_mpc_4.txt.json \
  results/uart_full_offset_0_count_100_mpc_4.txt \
  results/uart_full_offset_0_count_100_mpc_4.txt.json

./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 200 --dataset-offset 0 --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0000_count_0200_mpc_4.txt

./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 200 --dataset-offset 200 --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0200_count_0200_mpc_4.txt

./venv/bin/python scripts/run_renode_full_uart.py --dataset-count 100 --dataset-offset 400 --macs-per-cycle 4 --save-name chunked/renode_direct_uart_offset_0400_count_0100_mpc_4.txt

./venv/bin/python scripts/build_chunked_submission_artifacts.py

./venv/bin/python scripts/verify_benchmark_consistency.py
