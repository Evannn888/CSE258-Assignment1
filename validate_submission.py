"""Check submission row order, identifiers, targets, and reproducible file hashes."""
import argparse
import csv
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate(data_dir, output_dir):
    report = {}
    for task in range(1, 7):
        source = data_dir / f'task_{task}_test.csv'
        target = output_dir / f'task_{task}_predictions.csv'
        values = []
        count = 0
        with source.open(newline='') as src, target.open(newline='') as dst:
            expected, actual = csv.reader(src), csv.reader(dst)
            header = next(expected)
            if next(actual) != header:
                raise ValueError(f'Task {task}: header mismatch')
            for rowno, pair in enumerate(zip_longest(expected, actual), start=2):
                source_row, predicted_row = pair
                if source_row is None or predicted_row is None:
                    raise ValueError(f'Task {task}: row count mismatch at line {rowno}')
                if len(predicted_row) != len(header) or predicted_row[:-1] != source_row:
                    raise ValueError(f'Task {task}: identifiers/order mismatch at line {rowno}')
                value = predicted_row[-1]
                if task in (2, 4, 5):
                    if value not in source_row[-2:]:
                        raise ValueError(f'Task {task}: invalid candidate at line {rowno}')
                else:
                    value = float(value)
                    if (not math.isfinite(value) or value < 0
                            or (task in (1, 3) and not 1 <= value <= 5)):
                        raise ValueError(f'Task {task}: invalid numeric prediction at line {rowno}')
                    values.append(value)
                count += 1
        report[str(task)] = {'rows': count, 'sha256': digest(target)}
        if values:
            report[str(task)].update(min=min(values), max=max(values), mean=sum(values) / count)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('assignment1_2026'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    report = {'predictions': validate(args.data_dir, args.output_dir)}
    report['source_files'] = {
        name: digest(args.output_dir / name)
        for name in ('assignment1.py', 'writeup.txt', 'requirements.txt', 'validation_metrics.json')
        if (args.output_dir / name).is_file()
    }
    (args.output_dir / 'submission_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
