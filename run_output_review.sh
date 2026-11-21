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
if [[ "$mode" == --with-probes ]]; then
  # Uses saved frozen encoders and fits diagnostic linear probes only.
  # Existing analyze verifies code/dataset provenance before loading checkpoints.
  python -u run_transfer_study.py analyze --data-root "$data_root" --out "$study" --device cpu
fi
report="$PWD/output_reviews/review_$(date +%Y%m%d-%H%M%S)_$$"
python -u tools/review_transfer_outputs.py --data-root "$data_root" --study "$study" --out "$report"
printf '\nReport saved at %s\n' "$report"
