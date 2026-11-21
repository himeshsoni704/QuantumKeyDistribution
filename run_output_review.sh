#!/usr/bin/env bash
# Usage: bash run_output_review.sh DATA_ROOT STUDY_DIRECTORY [--with-probes]
set -euo pipefail
cd "$(dirname "$0")"
data_root="${1:?Pass the generated dataset directory, e.g. out_full_new}"
study="${2:-out_transfer_final/primary}"
mode="${3:-}"
if [[ -n "$mode" && "$mode" != --with-probes ]]; then
  echo 'Optional third argument is --with-probes' >&2
  exit 2
fi
# QKD_PYTHON can point to the exact interpreter used for training.
if [[ -n "${QKD_PYTHON:-}" ]]; then
  python_bin="$QKD_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
  python_bin="$PWD/.venv/bin/python"
elif [[ -x "$PWD/venv/bin/python" ]]; then
  python_bin="$PWD/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo 'Python was not found. Activate the project environment or set QKD_PYTHON.' >&2
  exit 127
fi
echo "Using Python: $python_bin"
"$python_bin" -c 'import numpy, pandas, scipy, sklearn, matplotlib' || {
  echo 'Required packages are missing. Activate the environment used for the completed study, or set QKD_PYTHON to its interpreter.' >&2
  exit 1
}
if [[ "$mode" == --with-probes ]]; then
  "$python_bin" -c 'import torch, joblib' || {
    echo 'Checkpoint probes require PyTorch and joblib in the selected environment.' >&2
    exit 1
  }
fi
if [[ "$mode" == --with-probes ]]; then
  # Uses saved frozen encoders and fits diagnostic linear probes only.
  # Existing analyze verifies code/dataset provenance before loading checkpoints.
  "$python_bin" -u run_transfer_study.py analyze --data-root "$data_root" --out "$study" --device cpu
fi
report="$PWD/output_reviews/review_$(date +%Y%m%d-%H%M%S)_$$"
"$python_bin" -u tools/review_transfer_outputs.py --data-root "$data_root" --study "$study" --out "$report"
printf '\nReport saved at %s\n' "$report"
