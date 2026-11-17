"""Assemble reviewed production modules into the single submission source.

Preserves the legacy implementation as reusable graph/data utilities. Production
modules receive independent names for their globals; base.foo resolves to the
legacy utility foo. No research files or caches are required by the result.
"""
import ast
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bindings(node):
    names = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split('.')[0])
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            names.update(x.id for x in ast.walk(target) if isinstance(x, ast.Name))
    return names


class Namespace(ast.NodeTransformer):
    def __init__(self, prefix, names):
        self.mapping = {name: prefix + name for name in names if name != 'base'}
        self.module_level = True

    def visit_Name(self, node):
        if node.id == 'base':
            raise ValueError('Standalone base reference must be an attribute')
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'base':
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Only global functions/classes are renamed, never instance methods.
        is_module = self.module_level
        self.module_level = False
        if is_module:
            node.name = self.mapping.get(node.name, node.name)
        node = self.generic_visit(node)
        self.module_level = is_module
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_alias(self, node):
        name = node.asname or node.name.split('.')[0]
        node.asname = self.mapping.get(name, name)
        return node

    def visit_Global(self, node):
        node.names = [self.mapping.get(n, n) for n in node.names]
        return node


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-source',type=Path,default=ROOT/'backups/pre_execution_20261109/assignment1.py')
    parser.add_argument('--ratings-source',type=Path,default=ROOT/'experiments/production_ratings.py')
    args=parser.parse_args()
    legacy_path = args.legacy_source
    production_paths={name:ROOT/f'experiments/production_{name}.py' for name in ['ratings','interactions','delta']}
    production_paths['ratings']=args.ratings_source
    legacy = ast.parse(legacy_path.read_text())
    # Keep only utilities actually used by the production pipelines/CLI. The
    # original task implementations stay in the backup, not the submission.
    utility_definitions = {}
    for node in legacy.body:
        for name in bindings(node):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                utility_definitions[name] = node
    needed = {'load','log','SEED','T_MIN_VALID','T_MAX_VALID'}
    for name in ['ratings','interactions','delta']:
        source = ast.parse(production_paths[name].read_text())
        needed.update(n.attr for n in ast.walk(source) if isinstance(n,ast.Attribute)
                      and isinstance(n.value,ast.Name) and n.value.id == 'base')
    pending = list(needed)
    while pending:
        name = pending.pop()
        node = utility_definitions.get(name)
        if node is None:
            continue
        for child in ast.walk(node):
            if isinstance(child,ast.Name) and child.id in utility_definitions and child.id not in needed:
                needed.add(child.id)
                pending.append(child.id)
    legacy.body = [n for n in legacy.body if isinstance(n,(ast.Import,ast.ImportFrom))
                   or bindings(n).intersection(needed)]
    sections = ['from __future__ import annotations\n',
                '# CSE258 A1: positive-conditional ratings, symmetric pair models, and terminal-time prediction.\n'
                '# Rebuilds all outputs from designated training CSVs; no research cache is required.\n',
                ast.unparse(legacy)]
    for name in ['ratings', 'interactions', 'delta']:
        path = production_paths[name]
        module = ast.parse(path.read_text())
        kept = []
        for node in module.body:
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                continue
            if isinstance(node, ast.Import):
                node.names = [a for a in node.names if a.name != 'assignment1']
                if not node.names:
                    continue
            if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            kept.append(node)
        module.body = kept
        names = set().union(*(bindings(n) for n in module.body))
        rewritten = Namespace(name + '_', names).visit(module)
        ast.fix_missing_locations(rewritten)
        sections.extend([f'\n# Production {name}: fixed validated configuration.\n', ast.unparse(rewritten)])
    sections.append((ROOT / 'experiments/production_cli.txt').read_text())
    target = ROOT / 'assignment1.py'
    target.write_text('\n\n'.join(sections) + '\n')
    compile(target.read_text(), str(target), 'exec')
    print(f'Built {target}: {len(target.read_text().splitlines())} lines')


if __name__ == '__main__':
    main()
