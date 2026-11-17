#!/bin/bash
# GPU experiment queue: one chain spec at a time through the v11 production function.
cd /data/UCSD/CSE258/A1
while read -r line; do
  [ -z "$line" ] && continue
  echo "$(date +%H:%M:%S) start $line" >> cache_v10/gpu_queue.log
  OMP_NUM_THREADS=4 venv/bin/python cache_v10/run_gchain13.py $line 2>&1 | grep -v "Warning\|sparse_csr\|^  return" >> cache_v10/gpu_queue.log
done < "$1"
echo "$(date +%H:%M:%S) QUEUE DONE $1" >> cache_v10/gpu_queue.log
