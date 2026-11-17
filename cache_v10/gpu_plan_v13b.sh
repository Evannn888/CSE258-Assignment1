#!/bin/bash
cd /data/UCSD/CSE258/A1
SL=200 SNNEG=16 SUF=_L200n16 OMP_NUM_THREADS=2 venv/bin/python cache_v10/sasrec_time.py 0,1,2,3,4 gap+month 20 64 >> cache_v10/sast_v13b.log 2>&1
SNNEG=64 SUF=_n64 OMP_NUM_THREADS=2 venv/bin/python cache_v10/sasrec_time.py 0,1,2,3,4 gap+month 20 64 >> cache_v10/sast_v13b.log 2>&1
echo "SAST V13B DONE" >> cache_v10/sast_v13b.log
