#!/usr/bin/env bash
set -euo pipefail
stage=${1:-core}
out=${2:-out_research}
case "$stage" in
core)
  python qkd_pipeline.py gen --out "$out" --hardware-effects
  python qkd_pipeline.py baselines --out "$out" --hardware-effects
  for spec in 'none noae_mean mean' 'full ae_lstm lstm' 'jepa jepa_lstm lstm' 'jepa jepa_transformer transformer'; do
    read -r features model backbone <<< "$spec"
    for command in train calibrate evaluate; do
      python qkd_pipeline.py "$command" --out "$out" --hardware-effects --feature-set "$features" --model-id "$model" --backbone "$backbone" --basic-evaluation
    done
  done
  ;;
shifts)
  python qkd_shift_audit.py --out "$out" --models noae_mean ae_lstm jepa_lstm jepa_transformer
  ;;
transfer)
  python qkd_pipeline.py transfer --out "$out" --hardware-effects --feature-set jepa
  ;;
*) echo 'Usage: bash run_research.sh {core|shifts|transfer} OUTPUT_DIRECTORY' >&2; exit 2;;
esac
