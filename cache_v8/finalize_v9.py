import csv, json, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np
root = Path('/data/UCSD/CSE258/A1')
SRC = {1: root / 'output/reproduce_v8', 3: root / 'output/reproduce_v8',
       2: root / 'output/reproduce_v9', 4: root / 'output/reproduce_v9', 5: root / 'output/reproduce_v9', 6: root / 'output/reproduce_v9'}
for t in range(1, 7):
    log = (SRC[t] / f'run_t{t}.log').read_text().splitlines()
    assert log[-1].strip() == 'exit 0', (t, log[-1])
def col(p):
    rows = list(csv.reader(open(p))); return rows[0], rows[1:]
for t in range(1, 7):
    _, new = col(SRC[t] / f'task_{t}_predictions.csv'); _, old = col(root / f'task_{t}_predictions.csv')
    assert [r[:-1] for r in new] == [r[:-1] for r in old]
    if t in (2, 4, 5):
        print(f'task {t}: agreement with v8 {np.mean([a[-1] == b[-1] for a, b in zip(new, old)]):.4f}')
    else:
        a = np.array([float(r[-1]) for r in new]); b = np.array([float(r[-1]) for r in old])
        print(f'task {t}: mean {a.mean():.6g} | v8 mean {b.mean():.6g} | identical {np.array_equal(a, b)} | corr(log) {np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]:.4f}')
metrics = {}
for t in range(1, 7):
    done = [l for l in (SRC[t] / f'run_t{t}.log').read_text().splitlines() if ' done: ' in l][-1]
    metrics.update({str(k): v for k, v in eval(done.split('done: ', 1)[1]).items()})
metrics['local_validation_summary'] = {
    '1': 'unchanged from v8 (EASE rating-neighbourhood features; v8 online 0.5151)',
    '2': 'fold0 0.9496 -> 0.9501 with the GBDT/MLP blend (two-fold mean 0.9500 -> 0.9506)',
    '3': 'unchanged from v8 (item + user EASE features; v8 online 0.6665)',
    '4': 'two-fold mean 0.9441 -> 0.9451 with the GBDT/MLP blend',
    '5': 'last-event acc 0.9344 -> 0.9352 with the 50/50 GBDT/MLP blend (large GBDT)',
    '6': 'Task-6-like problems (true final gaps): v8 7.518M/6.237M -> v9 7.500M/6.222M; last-event MAE 7.00M -> 6.98M'}
(root / 'v9/validation_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
B = root / 'backups/pre_v9_20261112'; B.mkdir(parents=True, exist_ok=True)
for f in ['assignment1.py', 'writeup.txt', 'README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'training.log'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]:
    shutil.copy2(root / f, B / f)
for name in ['assignment1.py', 'writeup.txt', 'README.md', 'validation_metrics.json', 'requirements.txt']:
    shutil.copy2(root / 'v9' / name, root / name)
for t in range(1, 7):
    shutil.copy2(SRC[t] / f'task_{t}_predictions.csv', root / f'task_{t}_predictions.csv')
(root / 'training.log').write_text(''.join((SRC[t] / f'run_t{t}.log').read_text() for t in (6, 5, 4, 2, 1, 3)))
print(subprocess.run([str(root / 'venv/bin/python'), 'validate_submission.py'], cwd=root, capture_output=True, text=True, check=True).stdout[:120])
sub = root / 'output/submission'
for f in sub.iterdir():
    f.unlink()
files = ['assignment1.py', 'writeup.txt'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]
for f in files + ['README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'validate_submission.py', 'training.log']:
    shutil.copy2(root / f, sub / f)
for zname in ['output/submission_v9.zip', 'output/submission.zip']:
    with zipfile.ZipFile(root / zname, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(root / f, arcname=f)
print('zip:', zipfile.ZipFile(root / 'output/submission_v9.zip').namelist())
