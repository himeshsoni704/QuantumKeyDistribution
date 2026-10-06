#!/usr/bin/env bash
# Start the QKD run in the background AND a watcher that copies data/ + plots/ into results/<run>/ on the v2_results branch and pushes every few minutes.
#   usage:  ./run_bg.sh [profile]          profile = quick | standard | large (default) | xl
# Needs: a Python venv with requirements installed, and git push access (SSH key or credential helper) for `origin`.
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-large}"
INTERVAL="${PUSH_EVERY_SECONDS:-600}"                   # push every 10 min
RESULTS_BRANCH="${RESULTS_BRANCH:-v2_results}"
[ -d .venv ] && source .venv/bin/activate || true

# results worktree: a separate checkout of the results branch (or this checkout, if you cloned that branch)
git fetch -q origin "$RESULTS_BRANCH"
if [ "$(git rev-parse --abbrev-ref HEAD)" = "$RESULTS_BRANCH" ]; then WT="$PWD"
else WT="$(cd .. && pwd)/qkd_results_wt"; [ -d "$WT" ] || git worktree add -q -B "$RESULTS_BRANCH" "$WT" "origin/$RESULTS_BRANCH"; fi
echo "$WT" > .results_wt; echo "$RESULTS_BRANCH" > .results_branch
echo "${PROFILE}-$(date +%Y%m%d-%H%M)" > .run_name
echo "profile=$PROFILE run=$(cat .run_name) start=$(date -Is)" > run_status.txt

QKD_PROFILE="$PROFILE" nohup python -u changed.py > run.log 2>&1 &
echo $! > run.pid
(   while kill -0 "$(cat run.pid)" 2>/dev/null; do sleep "$INTERVAL"; ./push_results.sh || true; done
    echo "finished=$(date -Is)" >> run_status.txt; ./push_results.sh || true
) > autopush.log 2>&1 &
echo $! > autopush.pid

echo "started: python PID $(cat run.pid), watcher PID $(cat autopush.pid); results -> origin/$RESULTS_BRANCH under results/$(cat .run_name)/"
echo "watch:   tail -f run.log   |   tail -f autopush.log   |   stop: ./stop_bg.sh"
