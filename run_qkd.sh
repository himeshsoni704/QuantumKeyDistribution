#!/usr/bin/env bash
set -Eeuo pipefail

cd ~/Downloads

echo "=== Activating environment ==="
source ~/pyclean/bin/activate

echo "=== Installing dependencies ==="
python -m pip install -r requirements.txt

echo "=== Running QKD script ==="
QKD_PROFILE=large python changed.py

echo "=== Checking git changes ==="
git status --short

echo "=== Committing ==="
git add -A
git commit -m "Auto-update QKD results $(date '+%Y-%m-%d %H:%M:%S')" || {
    echo "No changes to commit."
}

echo "=== Pushing ==="
git push origin claude

echo "=== DONE ==="
