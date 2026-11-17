"""Freeze reviewed v3 feature closures into a standalone production module."""
import ast,json
from pathlib import Path
from build_standalone import bindings
ROOT=Path(__file__).resolve().parents[1]


class Rename(ast.NodeTransformer):
    def __init__(self,names,prefix):
        self.names={name:prefix+name for name in names};self.top=True
    def visit_Name(self,n):
        if n.id=='a1':n.id='base'
        else:n.id=self.names.get(n.id,n.id)
        return n
    def visit_FunctionDef(self,n):
        top=self.top;self.top=False
        if top:n.name=self.names.get(n.name,n.name)
        n=self.generic_visit(n);self.top=top;return n
    visit_ClassDef=visit_FunctionDef


def closure(filename,roots,prefix):
    tree=ast.parse((ROOT/'experiments'/filename).read_text())
    definitions={name:n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom)) for name in bindings(n)}
    needed=set(roots);pending=list(needed)
    while pending:
        node=definitions[pending.pop()]
        for child in ast.walk(node):
            if isinstance(child,ast.Name) and child.id in definitions and child.id not in needed:
                needed.add(child.id);pending.append(child.id)
    tree.body=[n for n in tree.body if bindings(n)&needed]
    tree=Rename(needed,prefix).visit(tree);ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    config={task:json.loads((ROOT/f'experiments/v3_big_results_{task}.json').read_text())['selected'] for task in [1,3]}
    parts=['''"""V3: larger meta-training set and complementary conditional-rating features."""
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import assignment1 as base
''', 'V3_CONFIGS = '+repr(config)]
    for filename,roots,prefix in [
        ('v3_neighbors.py',['extract_topk_features'],'nk_'),
        ('v3_ordinal.py',['fit_ordinal','ordinal_features'],'ord_'),
        ('v3_temporal_production.py',['fit_features','extract_features'],'jt_'),
        ('v3_latent.py',['train','Model'],'lf_'),
        ('v3_task1_stages.py',['fit_stage_model','extract_stage_features'],'stg_'),
    ]:
        parts.append(closure(filename,roots,prefix))
    parts.append((ROOT/'experiments/v3_rating_entry.txt').read_text())
    path=ROOT/'experiments/production_ratings_v3.py'
    path.write_text('\n\n'.join(parts)+'\n')
    compile(path.read_text(),str(path),'exec')
    print(f'Assembled {path}, fixed configs {config}')


if __name__=='__main__':main()
