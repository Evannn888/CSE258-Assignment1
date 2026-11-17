#!/bin/bash
cd /data/UCSD/CSE258/A1 && venv/bin/python cache_v10/patch_v10_bfm.py && venv/bin/python cache_v10/patch_v10_t6.py && venv/bin/python cache_v10/patch_v10_b4r.py && venv/bin/python cache_v10/patch_v10_sas.py && venv/bin/python -c "import ast; ast.parse(open('v10/assignment1.py').read()); print('v10 syntax ok')"
