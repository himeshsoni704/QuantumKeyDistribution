"""Create a reviewable diff and a compact verification manifest."""
from pathlib import Path
import ast, difflib, hashlib, json

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
old = (work/'baseline_changed.py').read_text(encoding='utf-8')
new = (root/'changed.py').read_text(encoding='utf-8')
compile(new, str(root/'changed.py'), 'exec')
before = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
after = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
changed = [name for name in after if name in before and before[name]!=after[name]]
added = [name for name in after if name not in before]
(work/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/changed.py',tofile='after/changed.py')),encoding='utf-8')
checks=json.loads((work/'verification.json').read_text())
regression=json.loads((root/'documentation_work/review_test_results.json').read_text())
report=dict(code_sha256=hashlib.sha256((root/'changed.py').read_bytes()).hexdigest(),
            baseline_sha256=hashlib.sha256((work/'baseline_changed.py').read_bytes()).hexdigest(),
            functions_modified=changed,functions_added=added,focused_checks=checks['count'],
            existing_checks=regression['count'],synthetic_learnability_auc=checks['synthetic_auc'],
            diagnostic_results='benchmark_results.json',
            limitation='CPU checks and a small two-seed diagnostic only; expanded production study not run; gains mixed by protocol')
(work/'change_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
