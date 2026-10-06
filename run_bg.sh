#!/usr/bin/env bash
# Start the QKD run in the background AND a watcher that commits + pushes data/ and plots/ every few minutes.
#   usage:  ./run_bg.sh [profile]          profile = quick | standard | large (default) | xl
# Needs: a Python venv with requirements installed, and git push access (SSH key or credential helper) for `origin`.
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-large}"
INTERVAL="${PUSH_EVERY_SECONDS:-600}"                   # push every 10 min
BRANCH="results-${PROFILE}-$(date +%Y%m%d-%H%M)"
[ -d .venv ] && source .venv/bin/activate || true

git checkout -b "$BRANCH"                              # results go to their own branch, the code branch is never touched
touch .gitignore
grep -q 'data/\*.pkl' .gitignore || printf '\n# run artefacts that are too big / not results\ndata/*.pkl\nrun.log\nnohup.out\n' >> .gitignore

echo "profile=$PROFILE branch=$BRANCH start=$(date -Is)" > run_status.txt
QKD_PROFILE="$PROFILE" nohup python -u changed.py > run.log 2>&1 &
echo $! > run.pid

(   # watcher: runs while the python process is alive, then does one last push
    while kill -0 "$(cat run.pid)" 2>/dev/null; do
        sleep "$INTERVAL"
        ./push_results.sh "$BRANCH" || true
    done
    echo "finished=$(date -Is)" >> run_status.txt
    ./push_results.sh "$BRANCH" || true
) > autopush.log 2>&1 &
echo $! > autopush.pid

echo "started: python PID $(cat run.pid), watcher PID $(cat autopush.pid), results branch $BRANCH"
echo "watch:   tail -f run.log          (progress)   |   tail -f autopush.log  (pushes)"
echo "stop:    ./stop_bg.sh"
