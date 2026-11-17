import csv, json, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np
root = Path('/data/UCSD/CSE258/A1'); R = root / 'output/reproduce_v7'
for t in range(1, 7):
    log = (R / f'run_t{t}.log').read_text().splitlines()
    assert log[-1].strip() == 'exit 0', (t, log[-1])
def col(p):
    rows = list(csv.reader(open(p))); h = rows[0]; return h, rows[1:]
for t in range(1, 7):
    h, new = col(R / f'task_{t}_predictions.csv'); _, old = col(root / f'task_{t}_predictions.csv')
    assert [r[:-1] for r in new] == [r[:-1] for r in old]
    if t in (2, 4, 5):
        print(f'task {t}: agreement with v6 {np.mean([a[-1] == b[-1] for a, b in zip(new, old)]):.4f}')
    else:
        a = np.array([float(r[-1]) for r in new]); b = np.array([float(r[-1]) for r in old])
        print(f'task {t}: min {a.min():.4g} max {a.max():.4g} mean {a.mean():.6g} | v6 mean {b.mean():.6g} | corr(log) {np.corrcoef(np.log1p(a), np.log1p(b))[0, 1]:.4f}')
        assert np.isfinite(a).all() and (t == 6 or (a.min() >= 1 and a.max() <= 5))
metrics = {}
for t in range(1, 7):
    done = [l for l in (R / f'run_t{t}.log').read_text().splitlines() if ' done: ' in l][-1]
    metrics.update({str(k): v for k, v in eval(done.split('done: ', 1)[1]).items()})
metrics['local_validation_summary'] = {
    '1': 'fold-0 MSE: v6 GBDT 0.51943 -> GBDT/MLP 50/50 0.51789', '3': 'fold-0 MSE: v6 GBDT 0.65746 -> 50/50 0.65571',
    '4': 'fold0/fold1 acc: v6 0.9413/0.9395 -> +BERT4Rec 0.9426/0.9416', '5': 'last-event acc: v6 0.9322 -> +SASRec 0.9330',
    '6': 'per-bin scale calibrated on L2-L4 -> L1 (7.149M uniform 1.10 -> 7.137M per-bin)'}
(root / 'v7/validation_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
B = root / 'backups/pre_v7_20261111'; B.mkdir(parents=True, exist_ok=True)
for f in ['assignment1.py', 'writeup.txt', 'README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'training.log'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]:
    shutil.copy2(root / f, B / f)
for name in ['assignment1.py', 'writeup.txt', 'README.md', 'validation_metrics.json', 'requirements.txt']:
    shutil.copy2(root / 'v7' / name, root / name)
for t in range(1, 7):
    shutil.copy2(R / f'task_{t}_predictions.csv', root / f'task_{t}_predictions.csv')
(root / 'training.log').write_text(''.join((R / f'run_t{t}.log').read_text() for t in (6, 5, 4, 2, 1, 3)))
print(subprocess.run([str(root / 'venv/bin/python'), 'validate_submission.py'], cwd=root, capture_output=True, text=True, check=True).stdout[:120])
sub = root / 'output/submission'
for f in sub.iterdir():
    f.unlink()
files = ['assignment1.py', 'writeup.txt'] + [f'task_{t}_predictions.csv' for t in range(1, 7)]
for f in files + ['README.md', 'requirements.txt', 'validation_metrics.json', 'submission_manifest.json', 'validate_submission.py', 'training.log']:
    shutil.copy2(root / f, sub / f)
for zname in ['output/submission_v7.zip', 'output/submission.zip']:
    with zipfile.ZipFile(root / zname, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(root / f, arcname=f)
print('zip:', zipfile.ZipFile(root / 'output/submission_v7.zip').namelist())
