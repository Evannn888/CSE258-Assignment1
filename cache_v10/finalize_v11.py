"""Assemble v11: predictions from the v11 runs (other tasks from v10 - same code path), copy the
v11 files to the root (v10 backed up), validate, and write output/submission_v11.zip."""
import csv, json, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np
root = Path('/data/UCSD/CSE258/A1')
SRC = json.loads(sys.argv[1])                      # e.g. {"1": "output/reproduce_v10", ...}
SRC = {int(k): root / v for k, v in SRC.items()}
for t in range(1, 7):
    lines = (SRC[t] / f'run_t{t}.log').read_text().splitlines()
    assert any(' done: ' in l and f'{{{t}:' in l for l in lines), (t, 'no done line')
def col(p):
    rows = list(csv.reader(open(p))); return rows[0], rows[1:]
for t in range(1, 7):
    _, new = col(SRC[t] / f'task_{t}_predictions.csv'); _, old = col(root / f'task_{t}_predictions.csv')
    assert [r[:-1] for r in new] == [r[:-1] for r in old]
    if t in (2, 4, 5):
        print(f'task {t}: agreement with v10 {np.mean([a[-1] == b[-1] for a, b in zip(new, old)]):.4f}')
    else:
        a = np.array([float(r[-1]) for r in new]); b = np.array([float(r[-1]) for r in old])
        print(f'task {t}: mean {a.mean():.6g} | v10 mean {b.mean():.6g} | identical {np.array_equal(a, b)} | corr {np.corrcoef(a, b)[0, 1]:.4f}')
metrics = {}
for t in range(1, 7):
    done = [l for l in (SRC[t] / f'run_t{t}.log').read_text().splitlines() if ' done: ' in l][-1]
    metrics.update({str(k): v for k, v in eval(done.split('done: ', 1)[1]).items()})
metrics['local_validation_summary'] = json.loads((root / 'cache_v10/summary_v11.json').read_text())
(root / 'v11/validation_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
B = root / 'backups/pre_v11_20261115'; B.mkdir(parents=True, exist_ok=True)
for f in ['assignment1.py', 'writeup.txt', 'README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'training.log'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]:
    shutil.copy2(root / f, B / f)
for name in ['assignment1.py', 'writeup.txt', 'README.md', 'validation_metrics.json', 'requirements.txt']:
    shutil.copy2(root / 'v11' / name, root / name)
for t in range(1, 7):
    shutil.copy2(SRC[t] / f'task_{t}_predictions.csv', root / f'task_{t}_predictions.csv')
(root / 'training.log').write_text(''.join((SRC[t] / f'run_t{t}.log').read_text() for t in (6, 5, 4, 2, 1, 3)))
print(subprocess.run([str(root / 'venv/bin/python'), 'validate_submission.py'], cwd=root, capture_output=True, text=True, check=True).stdout[:200])
sub = root / 'output/submission'
sub.mkdir(exist_ok=True)
for f in sub.iterdir():
    f.unlink()
files = ['assignment1.py', 'writeup.txt'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]
for f in files + ['README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'validate_submission.py', 'training.log']:
    shutil.copy2(root / f, sub / f)
for zname in ['output/submission_v11.zip', 'output/submission.zip']:
    with zipfile.ZipFile(root / zname, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(root / f, arcname=f)
print('zip:', zipfile.ZipFile(root / 'output/submission_v11.zip').namelist())
