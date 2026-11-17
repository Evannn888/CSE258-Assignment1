#!/bin/bash
cd /data/UCSD/CSE258/A1
until grep -q "done:" output/reproduce_v12/run_t5.log; do sleep 30; done
B4SEED=100 B4TAG=_r2 OMP_NUM_THREADS=2 venv/bin/python cache_v10/b4r_time3.py 0,1,2,3 120 gap+month >> cache_v10/b4r_seeds.log 2>&1
B4SEED=200 B4TAG=_r3 OMP_NUM_THREADS=2 venv/bin/python cache_v10/b4r_time3.py 0,1,2,3 120 gap+month >> cache_v10/b4r_seeds.log 2>&1
echo "B4R SEEDS DONE" >> cache_v10/b4r_seeds.log
cache_v10/gpu_queue12.sh cache_v10/queue6.txt
