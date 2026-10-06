#!/usr/bin/env python3
"""Run changed.py with the `quick` profile (tiny sample sizes, one seed, one epoch) to check that every section executes on this machine.
No number produced by this run means anything. Outputs go to a scratch directory, so ./data and ./plots of a real run are not touched.

    python run_quick_test.py            # typically 30-60 min (the deep-learning sections dominate)
"""
import os, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
work = tempfile.mkdtemp(prefix='qkd_quick_')
print(f"running QKD_PROFILE=quick in {work}")
env = dict(os.environ, QKD_PROFILE='quick')
sys.exit(subprocess.call([sys.executable, os.path.join(here, 'changed.py')], cwd=work, env=env))
