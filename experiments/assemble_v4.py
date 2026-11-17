"""Append a frozen Task-1-only production override to the reviewed v3 source."""
import ast,json
from pathlib import Path
from assemble_v3_ratings import closure
from build_standalone import Namespace,bindings
ROOT=Path(__file__).resolve().parents[1]


def main():
    choice=json.loads((ROOT/'experiments/v4/fresh_raw_report.json').read_text())['selected']
    prefix='''import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import assignment1 as base
CHOICE = '''+repr(choice)+'\n'
    # Extract only the data-independent functions, removing all experimental I/O.
    helper=closure('v4_fresh_raw.py',['fit_reference','reference_features','fit_combiner','predict_combiner'],'')
    helper=helper.replace('old.StandardizedRidge','base.ratings_StandardizedRidge')
    module=ast.parse(prefix+'\n'+helper+'\n'+(ROOT/'experiments/v4_production_entry.txt').read_text())
    module.body=[n for n in module.body if not (isinstance(n,ast.Import) and any(a.name=='assignment1' for a in n.names))]
    names=set().union(*(bindings(n) for n in module.body))
    module=Namespace('v4_',names).visit(module);ast.fix_missing_locations(module)
    original=ast.parse((ROOT/'backups/pre_v4_20261109/assignment1.py').read_text())
    original.body=[n for n in original.body if not (isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))]
    main_fn=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    insertion=next(i for i,n in enumerate(main_fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
    main_fn.body[insertion:insertion]=ast.parse("parser.add_argument('--rating-workers',type=int,default=3,choices=[1,2,3])").body
    alias=ast.parse('_v4_previous_rating_task = ratings_production_rating_task')
    trailer=ast.parse('ratings_production_rating_task = v4_production_rating_task\nif __name__ == "__main__": main()')
    original.body.extend(alias.body+module.body+trailer.body)
    ast.fix_missing_locations(original)
    path=ROOT/'assignment1.py';path.write_text(ast.unparse(original)+'\n')
    compile(path.read_text(),str(path),'exec')
    print('Built v4 single file, choice:',choice)


if __name__=='__main__':main()
