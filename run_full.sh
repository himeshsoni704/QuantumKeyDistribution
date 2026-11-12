#!/usr/bin/env bash
# New transfer study only: never calls the legacy 'gen' or 'all' commands.
set -euo pipefail
cd "$(dirname "$0")"
data_root="${1:?Usage: bash run_full.sh EXISTING_OUT_OR_DATA NEW_RESULTS [plan|run]}"
results="${2:-out_transfer_final}"
action="${3:-run}"
if [[ "$action" != plan && "$action" != run ]]; then
  echo 'Third argument must be plan or run' >&2
  exit 2
fi
mkdir -p "$results"
common=(--data-root "$data_root" --epochs 100 --patience 10 --batch-size 32 --bootstrap 1000 --seeds 0 1 2 3 4)
run_stage() {
  local stage="$1"
  shift
  echo "Study stage: $stage"
  python -u run_transfer_study.py "$action" "${common[@]}" --out "$results/$stage" --resume "$@"
}
# Primary research comparisons. Source fits and scratch targets are reused.
run_stage primary --schemas common --normalizations standard \
  --models logistic rf boosting mlp ae_lstm jepa_lstm --representations latent \
  --regimes within zero_shot few_shot lopo joint --budgets 0 5 20 100 -1
run_stage qber --schemas qber --normalizations raw --models qber \
  --regimes within zero_shot lopo joint
# Representation and normalization ablations: no expensive few-shot Cartesian product.
run_stage representations --schemas common --normalizations raw standard honest \
  --models logistic rf mlp ae_lstm jepa_lstm --representations latent combined \
  --regimes within zero_shot lopo reference
run_stage full_features --schemas full specific --normalizations standard \
  --models logistic rf boosting mlp ae_lstm jepa_lstm --representations latent combined \
  --regimes within joint
run_stage support --schemas common_support --normalizations standard \
  --models logistic rf mlp ae_lstm jepa_lstm --representations latent \
  --regimes within zero_shot lopo
if [[ "$action" == run ]]; then
  python -u run_transfer_study.py analyze --data-root "$data_root" --out "$results/primary" --shap
  python -u run_transfer_study.py analyze --data-root "$data_root" --out "$results/full_features" --shap
fi
