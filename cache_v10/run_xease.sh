#!/bin/bash
# usage: run_xease.sh TASK MODE TAG THREADS "lams"
cd /data/UCSD/CSE258/A1
export OMP_NUM_THREADS=$4 NUMBA_NUM_THREADS=$4 NUMBA_CACHE_DIR=cache_v10/nbc_c
for f in $(seq 0 17); do
  venv/bin/python cache_v10/xease.py --task $1 --fold $f --mode $2 --lam $5 --tag $3 --threads $4 >> cache_v10/oof_$3_t$1.log 2>&1
done
echo "ALL DONE $3 t$1" >> cache_v10/oof_$3_t$1.log
