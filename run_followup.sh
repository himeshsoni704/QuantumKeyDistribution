#!/usr/bin/env bash
# Resumable, sequential research follow-up. Never modifies the historical study.
set -Eeuo pipefail
cd "$(dirname "$0")"
PY="${QKD_PYTHON:-$PWD/.venv/bin/python}"
OUT="${QKD_FOLLOWUP_OUT:-$PWD/out_followup_20261121}"
mkdir -p "$OUT" "$PWD/logs" "$OUT/.matplotlib"
export MPLCONFIGDIR="$OUT/.matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo 'This follow-up is already running.' >&2; exit 1; }
exec "$PY" -u tools/qkd_followup.py "${1:-all}" --data-root "$PWD/out_full_new" \
    --existing-study "$PWD/out_transfer_final" --out "$OUT" "${@:2}"
