#!/bin/bash
cd /data/UCSD/CSE258/A1
until grep -q "QUEUE DONE cache_v10/queue7.txt" cache_v10/gpu_queue.log; do sleep 30; done
SL=200 SUF=_L200 OMP_NUM_THREADS=2 venv/bin/python cache_v10/sasrec_time.py 0,1,2,3,4 gap+month 20 64 >> cache_v10/sast_v13.log 2>&1
SNNEG=16 SUF=_n16 OMP_NUM_THREADS=2 venv/bin/python cache_v10/sasrec_time.py 0,1,2,3,4 gap+month 20 64 >> cache_v10/sast_v13.log 2>&1
echo "SAST V13 DONE" >> cache_v10/sast_v13.log
