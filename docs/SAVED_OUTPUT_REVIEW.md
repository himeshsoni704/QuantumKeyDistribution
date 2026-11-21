# Evaluate the completed transfer study

The saved-output review creates score/QBER plots and per-attack tables from the
existing prediction files. MLP and Logistic Regression are the primary models
for this review; AE-LSTM and JEPA-LSTM remain included as secondary comparisons.
The report command does not fit detectors or choose a new threshold.

## Run on Himesh's computer

Activate the Python environment used for the completed run, then execute:

```bash
cd /home/himesh/Documents/QuantumKeyDistribution
git switch codex/standalone-qkd-dl
git pull --ff-only origin codex/standalone-qkd-dl
python -m unittest discover -s tests -p 'test_output_review.py' -v
bash run_output_review.sh out_full_new out_transfer_final/primary
```

`out_full_new` comes from the uploaded primary study's configuration. Substitute
its actual local location if it was moved. The script also accepts its `data/`
directory. If the study was moved, pass its actual location as the second argument.
Input data and saved predictions must describe the same generated arrays. Array
checksums and run/label joins are verified before calculating statistics.

To generate representation diagnostics from the saved checkpoints first:

```bash
bash run_output_review.sh out_full_new out_transfer_final/primary --with-probes
```

This invokes the existing `analyze` command with CPU inference, including its
provenance checks. Frozen encoders supply the embeddings. The diagnostic probes
are small fitted classifiers. This is additional analysis computation; the
existing QKD detectors are not retrained. Existing diagnostics may be reused or
refreshed according to the analyze command's diagnostic configuration. If a code
fingerprint mismatch is reported, use the study-compatible code checkout for
that analyze step instead of bypassing the check. Saved-output review itself
can read historical prediction files without loading a checkpoint.

A background run can be started with:

```bash
mkdir -p output_reviews
nohup bash run_output_review.sh out_full_new out_transfer_final/primary --with-probes \
  > output_reviews/review.log 2>&1 < /dev/null &
echo "$!" > output_reviews/review.pid
tail -f output_reviews/review.log
```

## Report contents

Each invocation creates a timestamped directory under `output_reviews/`.

- `plots/` has source-to-target plots for every saved zero-shot test evaluation
  of the selected models and seeds. The left panel compares honest/attack score
  distributions with the original source threshold. The right panel compares raw
  run QBER distributions. Classes are normalized independently, so their unequal
  counts do not dominate the visual comparison.
- `score_and_qber_summary.csv` records score quantiles, fraction over threshold,
  and both mean-window and pooled key QBER. A zero-comparison run is unavailable,
  rather than assigned QBER zero.
- `per_attack_by_seed.csv` gives IR, PNS, blocking, loss manipulation, E91 ancilla,
  depolarization and the BKM07 classes according to the actual stored class map.
  Each row includes run count, alerts and an exact binomial interval. The honest
  row reports false alarms; attack rows report recall.
- `per_attack_summary.csv` reports rates and variation across model seeds.
  These seeds reuse the same held-out runs. Counts are not multiplied by five
  to construct a narrower statistical interval.
- `paired_gains_by_seed.csv` keeps the saved paired bootstrap intervals and
  places transfer recall/FAR next to scratch recall/FAR for every gain.
- `paired_gain_consistency.csv` counts positive-gain seeds and seeds with an
  entirely positive paired interval. This is descriptive consistency across
  model seeds, not independent evidence from five new physical datasets.
- `representation_diagnostics.csv` summarizes existing attack probes, protocol
  identity probes and class-conditional centroid comparisons.
  `missing_diagnostics.csv` lists absent files. Missing files are not evidence
  of either good or bad representations.
- `evaluation_inventory.csv`, `report.json` and `README.md` record report coverage,
  selected models and input fingerprints.

The raw QBER figure uses the mean of valid-window key QBER values, matching the
QBER baseline's aggregation. The CSV also exports pooled QBER, calculated as
all errors divided by all eligible comparisons. They differ when window support
varies. Neither plotted QBER is standardized.

## Interpretation

A target with separated honest/attack scores but nearly all scores above the
source threshold suggests threshold or baseline transfer failure. Overlapping
score distributions suggest limited discrimination in the selected features.
Both conditions can occur together. Do not retune on the final target test set.

Inspect which attacks dominate the misses before changing the architecture.
For example, low BB84 recall could combine successful IR detection with missed
low-disturbance loss/PNS scenarios. Saved class labels identify simulator
mechanisms, not a diagnosis of real hardware.

Protocol-identity probe accuracy can be high while useful attack information is
also present. Compare source-to-target attack-probe performance, raw-feature
baselines and the existing latent/combined ablations. The current code does not
save every embedding to disk, so a new geometric analysis requires checkpoint
inference. Saved centroid metrics alone do not establish representation semantics.

The script prioritizes evaluation of completed outputs. The separate focused
training design—MLP/Logistic Regression with declared target honest-reference
and threshold-calibration budgets—requires disjoint reference/calibration/test
groups and should follow this diagnosis. No new detector training is launched
by the saved-output review.
