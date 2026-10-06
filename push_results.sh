#!/usr/bin/env bash
# Commit and push everything the run has produced so far (CSV / JSON / PNG + the last lines of the log).
cd "$(dirname "$0")"
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
tail -n 3000 run.log > run_tail.log 2>/dev/null || true
git add -f data/*.csv data/*.json plots/*.png run_tail.log run_status.txt .gitignore 2>/dev/null || true
if ! git diff --cached --quiet; then
    git commit -q -m "run results $(date -Is)"
fi
for t in 1 2 3 4; do git push -q -u origin "$BRANCH" && break || sleep $((2 ** t)); done
