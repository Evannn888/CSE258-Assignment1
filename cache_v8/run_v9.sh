#!/bin/bash
# usage: run_v8.sh <task> [<task> ...]   (each task in its own process, log ends with "exit <code>")
cd /data/UCSD/CSE258/A1/v9
for t in "$@"; do
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 ../venv/bin/python assignment1.py \
    --data-dir ../assignment1_2026 --output-dir ../output/reproduce_v9 --tasks $t \
    > ../output/reproduce_v9/run_t$t.log 2> ../output/reproduce_v9/stderr_t$t.log
  echo "exit $?" >> ../output/reproduce_v9/run_t$t.log
done
