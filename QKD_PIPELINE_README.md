# Standalone QKD deep-learning pipeline

`qkd_pipeline.py` contains the simulator, feature extraction, datasets, neural
models, training, calibration, evaluation and experiment commands. It does not
import `changed.py`, execute a notebook, or require another project Python file.
The numbered section banners make it possible to read one part at a time.

## Install and first run

Python 3.13 is recommended. Install the supplied repository's pinned requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python qkd_pipeline.py selftest
```

For a copy of the standalone file outside this repository, its required packages
are `numpy pandas scipy scikit-learn torch joblib`; XGBoost is optional. The
scientific-package versions tested are recorded in `requirements.txt`.
Matplotlib, SHAP and Jupyter are not required by this script. CPU execution is
supported; `--device auto` selects CUDA when available.

Start with staged smoke commands, all using the same data/configuration flags:

```bash
python qkd_pipeline.py gen       --smoke --out out_smoke
python qkd_pipeline.py train     --smoke --out out_smoke
python qkd_pipeline.py calibrate --smoke --out out_smoke
python qkd_pipeline.py evaluate  --smoke --out out_smoke
```

This trains the default autoencoder + LSTM for all three protocols and two seeds.
Use `--protocols bb84`, `--seeds 0`, or `--basic-evaluation` to narrow a development
run. `--basic-evaluation` omits prefix-delay and occlusion passes. A smoke run is an
execution check, not evidence of scientific accuracy or calibrated deployment
performance. Even `all --smoke` trains many models; staged commands are faster for
iteration.

```bash
# Compare temporal models on the SAME cached data.
python qkd_pipeline.py train --smoke --out out_smoke --backbone transformer
python qkd_pipeline.py calibrate --smoke --out out_smoke --backbone transformer
python qkd_pipeline.py evaluate --smoke --out out_smoke --backbone transformer

# No-autoencoder control; model id defaults to noae_lstm.
python qkd_pipeline.py train --smoke --out out_smoke --feature-set none
python qkd_pipeline.py calibrate --smoke --out out_smoke --feature-set none
python qkd_pipeline.py evaluate --smoke --out out_smoke --feature-set none

# Optional experiment suites using existing cached data.
python qkd_pipeline.py baselines    --smoke --out out_smoke
python qkd_pipeline.py sweep_latent --smoke --out out_smoke
python qkd_pipeline.py transfer     --smoke --out out_smoke --backbone lstm
python qkd_pipeline.py joint        --smoke --out out_smoke
python qkd_pipeline.py compact      --smoke --out out_smoke

# Full tiered study; choose a NEW output directory. This is substantial compute.
python qkd_pipeline.py all --out out_full
```

`baselines` fits, calibrates and evaluates QBER, logistic regression, validation-tuned
random forest/boosting, and a summary MLP. Neural model ids belong to their own
`train`/`calibrate`/`evaluate` commands. Use `--model-id` to keep custom experiments
separate. `--feature-set latent`, `latent_score`, and `full` implement the component
ablations; `none` skips honest pretraining. Use `--train_shuffle` for shuffled-order
training. `--no_crossfit` is an explicit, optimistic residual-feature ablation.

For a held-out attack study, generate a fresh dataset with, for example,
`--holdout_attack pns --protocols bb84`, and repeat those flags in subsequent
commands. That family has label -1, is absent from training/validation/calibration
and the attack head, and appears only in test/stress anomaly evaluation.

For diagnosis, choose an integer test/stress index from
`data/bb84/splits.json` and run:

```bash
python qkd_pipeline.py diagnose --smoke --out out_smoke \
  --protocols bb84 --seeds 0 --run-index <TEST_INDEX>
```

The JSON reports calibrated class probabilities, NORMAL/UNCERTAIN/ATTACK, an
independent anomaly flag, the most anomalous windows and signed feature residuals.
Missing observations return `insufficient_observations`. No-autoencoder models
report anomaly fields as unavailable.

## What is preserved from the pasted code

The original simulator and classical extractor definitions are embedded directly,
with formatting changes only. This includes GYS channel constants, decoy settings,
PNS/blocking/loss mechanisms, BKM07 forward/return and SIFT/CTRL rules, exact-K
continuation and truncation, E91 density-matrix sampling, setting order, CHSH signs,
and the singlet key-bit convention. The original attack class lists are retained;
`clean` is displayed as `honest` at class index zero.

The default is K=2,000 usable key events, BB84 distance 0–100 km, BKM07 distance
0–15 km, and E91 visibility 0.85–0.99. Honest drift is label-independent. The
source's exact extractors run during generation; their returned values are saved
under `original_features/` for audit. Those completed-session vectors are **not**
repeated into every window.

The new brief's neural input is built from observable retained raw records:
50 contiguous emitted-round windows, including failed detections; local counts
and rates; missingness masks; supporting counts. BKM07 comparisons use the exact
branch-specific bits. E91 detector/gain features remain unavailable. CHSH needs
all four setting pairs. Supporting count for the derived CHSH statistic is the
minimum of those four counts; CHSH is not represented as a binomial error rate.

Original aggregate features such as temporal dispersion and decoy bounds remain
available in the audit files. The new DL feature list is the brief's local-rate
union schema, printed in `data/feature_schema.json`, rather than the original
8-channel detected-event encoder. All simulated comparison bits are treated as
disclosed test bits; an operational QKD system must sacrifice disclosed key bits.

## Training and leakage guards

- Groups, not windows, are split 70/10/5/5/10 into train/val/cal_A/cal_B/test.
  Exact integer allocations are recorded. Stress groups are entirely separate.
- Normalization uses only valid training entries, per protocol and feature.
- `RunDataset` and `DataBundle` have no evaluation-metadata field. Model/training
  functions do not open `eval_meta.json`.
- Honest pretraining uses honest train data; honest validation chooses the epoch.
  Five fold AEs produce excluded-group classifier training features. A separately
  fitted full AE supplies validation, calibration, test and deployment features.
  Recorded fold provenance and cached features are saved beside each checkpoint.
- Every neural phase uses AdamW, a 100-epoch cap, patience 10 and minimum validation
  improvement 1e-4, restoring the best checkpoint. Smoke mode uses three epochs.
- Temperature and anomaly thresholds use cal_A; the abstention band uses cal_B.
  Test evaluation refuses missing thresholds or changed checkpoint hashes.
- Reported false-alert/miss rates use honest/attack populations as their respective
  denominators, with strict `>`/`<` comparisons. Threshold guarantees are empirical
  on calibration samples, not confidence guarantees on new data. Smoke calibration
  subsets are deliberately tiny.

## Defaults and deliberate deviations from the proposed brief

1. The original exact-K setup is preserved instead of replacing it with a fixed
   20,000-emission synthetic generator. No synthetic physics fallback is used.
   Smoke mode changes K to 64, uses 40 main sessions/class, two stress sessions/class,
   three epochs and seeds 0/1. Full mode uses 300 main and 30 stress sessions/class,
   five seeds, and up to 100 epochs. Class counts are an explicit configuration choice.
2. New 50-window emitted-round features replace the original DL event windows, as
   requested by the second pasted brief. Original classical extraction is retained
   separately. This is a new DL study, not a claim of identical notebook predictions.
3. The simulator does not expose exact attack-hit times or a delayed/short-intervention
   API. Native bursty intercept attacks are retained; no new attack schedule is
   invented. Delay is measured from mechanism enablement at round zero and is labeled
   accordingly. Controlled delayed/short-attack studies are not implemented.
4. To resolve the brief's conflicting instructions about latent selection, select
   latent size on honest validation reconstruction only. Unseen stress data are
   evaluated afterward, never used to select a latent size.
5. Compact settings are compared on validation. The selected combination is retrained,
   recalibrated and tested; isolated compact variants are not selected using test scores.
   Short-attack-specific compact claims cannot be made without the simulator extension
   described above. Weak-attack stress performance is available for the combined model.
6. The shared-input transfer variant uses `qber_key`, the common observable with a
   consistent definition across all three protocols. Its supporting count remains in
   the input. No optical feature is treated as comparable to an unmodeled E91 detector.
7. Pairwise transfer runs the requested `--backbone`; run the other backbone explicitly
   if desired. Budgets larger than available class support are logged as unavailable,
   never silently substituted. Validation and calibration labels are counted separately.
8. Full research-scale results and GPU behavior require their own runs. Runtime,
   accuracy and significance are never hard-coded. Test-group bootstrap intervals
   currently summarize AE-versus-no-AE macro-F1 differences for each backbone/seed;
   transfer tables retain all paired seed-level gains, including negative gains.

## Output layout

```text
out/
  data/{protocol}/
    records/*.npz             # Observable per-round arrays only
    original_features/*.json  # Original extractor audit values, not model inputs
    arrays.npz                # Numerators, denominators, emitted-round edges
    runs.csv                  # Run/group ids, seed, label
    eval_meta.json            # Attack/noise metadata: evaluation only
    normalizer.json, splits.json, config.json
  models/{protocol}/{model_id}/seed{s}/
    best.pt, history.json, crossfit_audit.json, train_features.pt
  results/{protocol}/{model_id}/seed{s}.json
    # Classification, calibration, selective, delay and occlusion reports
  results/{protocol}/{model_id}/seed{s}_{test|stress}_predictions.npz
  tables/
    main.csv, main_summary.csv, paired_bootstrap.json
    transfer.csv, transfer.json, latent_validation.csv, compact.csv
  locked_thresholds.json
  diagnosis.json
  invocation_config.json
  run_log.jsonl
```

Each source trace is an independent group; if you add copies or augmentations,
retain the original group id. Generated records can consume considerable disk
space, particularly for long lossy exact-K sessions. Keep each dataset under a
fresh output directory. Do not deserialize model files from untrusted sources.

## Validation performed

The embedded self-test passed 29 checks. An additional source comparison verified
that all 40 embedded simulator/helper function and class definitions have identical
Python syntax trees to the pasted source (formatting aside). CPU smoke validation
generated 504 sessions across the three protocols and exercised the comparison
matrix, latent sweep, all six transfer directions, joint training, and compact
experiments with two seeds. A separate E91 ancilla holdout run verified exclusion
from training/calibration, unknown-attack reporting, diagnosis, and empty-input
handling. These runs validate execution, not full-scale research conclusions.
