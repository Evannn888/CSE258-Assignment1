#!/bin/bash
# usage: run_oof.sh TASK TAG NPROC "extra args"
cd /data/UCSD/CSE258/A1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
seq 0 17 | xargs -P $3 -I{} sh -c "venv/bin/python cache_v10/bfm_oof.py --task $1 --tag $2 --fold {} $4 >> cache_v10/oof_$2_t$1.log 2>&1"
echo "ALL DONE $2 t$1" >> cache_v10/oof_$2_t$1.log
