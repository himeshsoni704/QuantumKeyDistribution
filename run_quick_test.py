#!/usr/bin/env python3
"""Run changed.py at tiny scale to check that every section executes on this machine (no numbers from this run mean anything).

The full-scale constants in changed.py are rewritten with the text substitutions in quick_test_subs.json (fewer runs, seeds, epochs,
sessions) and the result is executed in a scratch directory, so ./data and ./plots of a real run are not touched.

    python run_quick_test.py            # typically 30-60 min on a laptop (the deep-learning sections dominate)
"""
import json, os, subprocess, sys, tempfile

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'changed.py'), encoding='utf-8').read()
for old, new in json.load(open(os.path.join(here, 'quick_test_subs.json'))):
    if old not in src:
        print(f"WARNING: substitution target not found (script changed?): {old!r}")
    src = src.replace(old, new)
work = tempfile.mkdtemp(prefix='qkd_quick_')
path = os.path.join(work, 'changed_quick.py')
open(path, 'w', encoding='utf-8').write(src)
print(f"running the quick configuration in {work}")
sys.exit(subprocess.call([sys.executable, path], cwd=work))
