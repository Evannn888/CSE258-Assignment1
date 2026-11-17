import csv, json, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np
root = Path('/data/UCSD/CSE258/A1'); R = root / 'output/reproduce_v8'
for t in range(1, 7):
    log = (R / f'run_t{t}.log').read_text().splitlines()
    assert log[-1].strip() == 'exit 0', (t, log[-1])
def col(p):
    rows = list(csv.reader(open(p))); h = rows[0]; return h, rows[1:]
for t in range(1, 7):
    h, new = col(R / f'task_{t}_predictions.csv'); _, old = col(root / f'task_{t}_predictions.csv')
    assert [r[:-1] for r in new] == [r[:-1] for r in old]
    if t in (2, 4, 5):
        print(f'task {t}: agreement with v7 {np.mean([a[-1] == b[-1] for a, b in zip(new, old)]):.4f}')
    else:
        a = np.array([float(r[-1]) for r in new]); b = np.array([float(r[-1]) for r in old])
        print(f'task {t}: min {a.min():.4g} max {a.max():.4g} mean {a.mean():.6g} | v7 mean {b.mean():.6g} | corr(log) {np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]:.4f}')
        assert np.isfinite(a).all() and (t == 6 or (a.min() >= 1 and a.max() <= 5))
metrics = {}
for t in range(1, 7):
    done = [l for l in (R / f'run_t{t}.log').read_text().splitlines() if ' done: ' in l][-1]
    metrics.update({str(k): v for k, v in eval(done.split('done: ', 1)[1]).items()})
metrics['local_validation_summary'] = {
    '1': 'fold-0 MSE with a smaller GBDT stacker: 0.52710 -> 0.52472 with EASE rating-neighbourhood features',
    '2': 'fold0/fold1 acc: 0.9471/0.9484 -> 0.9496/0.9504 with EASE (lambda 300)',
    '3': 'fold-0 MSE with a smaller GBDT stacker: 0.66399 -> 0.66113 with EASE rating-neighbourhood features',
    '4': 'fold0/fold1 acc: 0.9426/0.9416 -> 0.9445/0.9436 with EASE (lambda 500 + 2000)',
    '5': 'last-event acc: 0.9319 -> 0.9334 with the larger pair classifier',
    '6': 'Task-6-like problems from the Task 3/4 and 1/2 files (true final gaps): MAE 7.683M/6.375M -> 7.518M/6.237M'}
(root / 'v8/validation_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
B = root / 'backups/pre_v8_20261112'; B.mkdir(parents=True, exist_ok=True)
for f in ['assignment1.py', 'writeup.txt', 'README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'training.log'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]:
    shutil.copy2(root / f, B / f)
for name in ['assignment1.py', 'writeup.txt', 'README.md', 'validation_metrics.json', 'requirements.txt']:
    shutil.copy2(root / 'v8' / name, root / name)
for t in range(1, 7):
    shutil.copy2(R / f'task_{t}_predictions.csv', root / f'task_{t}_predictions.csv')
(root / 'training.log').write_text(''.join((R / f'run_t{t}.log').read_text() for t in (6, 5, 4, 2, 1, 3)))
print(subprocess.run([str(root / 'venv/bin/python'), 'validate_submission.py'], cwd=root, capture_output=True, text=True, check=True).stdout[:160])
sub = root / 'output/submission'
for f in sub.iterdir():
    f.unlink()
files = ['assignment1.py', 'writeup.txt'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]
for f in files + ['README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'validate_submission.py', 'training.log']:
    shutil.copy2(root / f, sub / f)
for zname in ['output/submission_v8.zip', 'output/submission.zip']:
    with zipfile.ZipFile(root / zname, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(root / f, arcname=f)
print('zip:', zipfile.ZipFile(root / 'output/submission_v8.zip').namelist())
