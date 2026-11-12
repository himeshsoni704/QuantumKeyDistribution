# Cross-protocol transfer study

This runner answers a binary simulation question: can a detector fitted on one
protocol recognize attacked runs from another? It does not certify secret-key
security. The existing `qkd_pipeline.py`, simulators, legacy multiclass experiments,
`qkd_shift_audit.py`, datasets and results are unchanged.

## Run on Linux / RTX 4090

Use the Python environment that already ran the pipeline. The new runner uses
NumPy, pandas, SciPy, scikit-learn, PyTorch, joblib and matplotlib. SHAP is optional.
No new large pretrained models are downloaded. Defaults are batch size 32, small
32-unit networks, latent width 8, at most **100 epochs**, patience 10, AdamW and
norm-5 gradient clipping. Entire datasets remain on CPU. These sizes also target
4 GB GPUs, but actual peak memory must be measured on the chosen environment.

```bash
cd ~/Documents/QuantumKeyDistribution
git pull --ff-only origin codex/standalone-qkd-dl
python -m unittest discover -s tests -p 'test_transfer_study.py' -v
# Replace out with the directory containing your EXISTING data/ directory.
bash run_full.sh out out_transfer_final plan
bash run_full.sh out out_transfer_final run 2>&1 | tee transfer_run.log
```

Rerun the same command after an interruption. Completed source fits, scratch
models and evaluations are reused; a training phase interrupted before its best
checkpoint is saved restarts that phase. This is job-level resumption, not
mid-epoch optimizer-state resumption. Do not change code/configuration halfway
through a study. A changed fingerprint requires a new output directory.

`run_full.sh` is a **new** transfer-study launcher, not the legacy all command.
It never generates the full dataset. The primary stage is followed by bounded
representation/normalization, full-feature and support ablations. It runs only
latent representations for primary few-shot comparisons to avoid a huge grid.
The primary source checkpoint diagnostics run last. Optional simulation shifts
are separate, explicitly requested commands below.

The plan prints fit count, input-array RAM and a conservative artifact allowance.
It does not invent an hours estimate. Source training durations are saved in
`training.json`. Use a small one-seed job to benchmark before committing to the
full matrix. Tree fits, bootstrap and SHAP use CPU; a GPU cannot accelerate all
stages. Five seeds quantify initialization/optimization variability on the saved
split, not five independently generated QKD datasets.

### Small first run (existing real data, not synthetic performance)

```bash
python -u run_transfer_study.py run --data-root out --out out_transfer_pilot \
  --schemas common --normalizations standard --models logistic mlp ae_lstm jepa_lstm \
  --representations latent --regimes within zero_shot lopo \
  --seeds 0 --epochs 5 --bootstrap 100 --resume
```

The pilot intentionally differs from the final 100-epoch cap and must have its
own output directory. Do not compare a pilot checkpoint to final models as if
training budgets were identical.

## Data and scientific provenance

The loader reads only feature_schema.json, config.json, arrays.npz, runs.csv,
splits.json and (if present) audit_manifest.json. It does not read raw records or
copy the 28 GB dataset. Arrays are verified against their manifest, when present;
metadata and arrays are hashed into the new study manifest. Historical data
without an audit manifest are explicitly marked as such. This does not prove
historical physical correctness. The original simulator hash is retained even
when it differs from today's code; no claim of applying later physics fixes is
made. Unknown attack label -1 maps to binary attack, not honest.

One row must be one independent simulated session. Split overlap is rejected.
Exact duplicate observations are counted, including across splits: identical
low-count statistics can occur by chance, so that count is a diagnostic rather
than an automatic assertion of leakage. The loader cannot prove independent
random streams from arrays alone. It preserves configuration/run metadata for
that audit. Existing full datasets are neither overwritten nor regenerated.

## Experiment contracts

| Regime | Fitting and model selection | Calibration | Target exposure |
|---|---|---|---|
| Within | Training and validation of one protocol | Its cal_A/cal_B | Independent test |
| Strict zero-shot | Source training and validation only | Source only | Final evaluation |
| LOPO | The two source protocols only | Those sources only | Held-out protocol evaluation |
| Few-shot | Matched target groups for all methods; source initialization for transfer | Target cal_A/cal_B | Explicit supervised adaptation |
| Joint shared binary | All three protocols, shared mapping and binary head | Pooled cal_A/cal_B | Seen-protocol independent tests |
| Honest reference | Source model and source threshold, target cal_A honest preprocessing reference | No target attack labels | Reference adaptation, NOT zero-shot |

The legacy joint experiment remains available and is a different architecture:
shared modules with protocol-specific adapters/heads/calibration. The new joint
binary experiment uses a single mapping/head and a pooled calibration mixture.
Source mixtures in LOPO/joint are weighted by their available run counts, with
binary-balanced training loss; no equal per-protocol FAR guarantee is claimed.

Strict transfer uses the same adapter, encoder, downstream weights, normalizer,
temperature and threshold. No random target adapter is substituted. The loader
can read target labels, but fitting receives **only source bundles**; assertions
check group provenance before reporting strict results. Tests mutate target
labels and verify identical source parameters and scores.

Few-shot budgets are 0, 5, 20, 100 and all (`-1`) runs **per original simulator
class**, with binary training labels. Thus total labeled budgets differ with the
number of attack classes; use reported run counts for label-efficiency analysis.
Zero uses strict transfer, never an empty supervised fit. Scratch starts fresh.
Head-only copies the complete source model/preprocessor and reinitializes only
the binary head. Partial also trains the LSTM readout; for the MLP, partial and
head-only coincide because it has no temporal module. The source window mapping
and AE/JEPA encoder stay frozen. Target honest representation pretraining is
used only by scratch, on the selected honest budget. Source transfer does not
quietly train a target adapter. Full target validation and calibration are
reported separately; a five-per-class result does NOT mean only five labels
were used across the entire research procedure.

Full/specific few-shot mapping is an optional union-schema diagnostic using
named feature coordinates, including potentially source-unobserved coordinates;
it is not zero-shot. The default full-feature stage stays within protocol/joint.

## Features

| Schema | Measurements | Scope |
|---|---|---|
| qber | Window key disagreement rate | Minimal portable baseline |
| common | Key disagreement, adjacent change, absolute adjacent change | Portable error dynamics |
| common_support | Common plus log(1 + disclosed comparisons) | Tests support/shot-noise information and shortcuts |
| full | Original observable union, except duplicate qber_zsr | Within/joint and optional adapted diagnostic |
| specific | Full excluding qber_key | Non-common feature ablation, not a pure protocol-identity measure |

Error means disclosed key comparison mismatch, with Bob's E91 singlet result
flipped as in the simulator. This is comparable operational error, not identical
underlying physics. Adjacent changes are masked unless both windows contain
comparisons. Unavailable values are zero after standardization and accompanied
by availability masks. Missingness can reveal protocol identity in full inputs.
The QBER-only threshold baseline uses the mean of available window QBERs, not a
pooled photon-weighted error ratio; the latter would be a separate baseline.

Detection_rate, accepted_fraction and CHSH are not silently renamed as equivalent
cross-protocol observables. They remain in the full schema. Full and specific
schemas are excluded from strict zero-shot/LOPO. Adding optical common features
requires a separately reviewed denominator/opportunity definition.

All windows are fractions of a completed session's emitted slots. This is an
offline monitoring study, not a validated real-time alarm system.

## Representations and normalization

MLP: per-window linear/ReLU mapping and masked mean readout. LSTM: same mapping,
small unidirectional temporal readout and masked mean. AE and JEPA use the same
small per-window shared input encoder. AE learns masked honest reconstruction.
JEPA predicts masked teacher window embeddings with a causal GRU predictor,
EMA teacher (0.99) and variance regularization. The predictor is discarded at
inference. The encoder alone is not a context-aware temporal encoder. This is a
JEPA-inspired ablation, not a claim of semantic understanding or a faithful
reimplementation of image JEPA.

Honest-only pretraining uses train and honest validation groups. The selected
encoder is frozen for downstream classification, avoiding independently trained
fold encoders with incompatible latent coordinates. `latent` is genuinely
latent-only; `combined` concatenates raw standardized values, masks and latent
vectors. Raw models are separate baselines. This implementation does not add
reconstruction residuals to combined inputs; legacy full-AE experiments remain
available for that additional ablation.

`raw` applies no centering/scaling. `standard` fits source training observations.
`honest` fits source honest training observations. None uses target statistics
in strict transfer. The separate honest-reference experiment substitutes target
honest cal_A reference statistics but leaves source temperature/threshold fixed;
its success and false alarms must be measured, not assumed.

## Calibration, metrics and statistics

Temperature scaling uses cal_A labels (except the QBER baseline). The alarm
threshold uses only honest cal_B scores at the conformal order statistic
ceil((n+1)*(1-alpha)), with strict score > threshold. If too few honest calibration
runs exist, threshold=1 produces no alerts and insufficient_resolution=true.
This is a source-exchangeability marginal reference, NOT a distribution-shift
false-alarm guarantee. Old tiny calibration sets may have no useful resolution.

Metrics: binary macro-F1 at that threshold, recall, honest FAR, accuracy, AUROC,
average precision (not trapezoidal PR area), balanced-prior AP, Brier, ten-bin ECE
and reliability bins. Report natural attack prevalence. Exact binomial intervals
are saved for recall/FAR; bootstrap intervals resample full run groups. Paired
transfer gains resample the same target groups. Bootstrap intervals are
conditional on the saved datasets; seeds are not additional independent test
runs. Summary standard deviation is across configured model seeds.

Outputs include predictions, thresholds and group IDs; training groups and label
usage; preprocessing; history; trusted local joblib checkpoints and hashes;
code/dataset fingerprints; result/summary CSVs; paired gains; transfer gaps;
directional asymmetry; label-efficiency curves as data; negative-transfer
frequencies; and zero-shot AUROC matrices. Negative-transfer frequencies are
reported both by point estimate and by wholly negative paired intervals.
Multiple-comparison correction is not applied; treat these as exploratory
comparisons and preselect primary hypotheses before final significance claims.

## Representation and explanation diagnostics

```bash
python run_transfer_study.py analyze --data-root out --out out_transfer_final/primary --shap
```

Frozen source attack probes are trained only on source training representations
and evaluated across protocols. A separate protocol probe uses all protocols'
training groups, balanced by protocol and binary state, with held-out test groups.
That post-hoc probe is not part of zero-shot fitting. Attack-strength matching is
not performed automatically; binary balancing alone cannot remove every nuisance.

Class-conditional standardized centroid distances use source-training scaling.
PCA is source fitted. These unpaired comparisons do not misuse CKA across
unrelated sessions. Saved PCA coordinates permit plotting; no t-SNE plot is
used as evidence of invariance.

TreeSHAP uses source-training background for RF/compatible boosting estimators,
with summary/dependence plots and saved per-run SHAP values/labels for honest and
attack analysis. Source-target importance ranking correlations use the same
feature schema. Missing/incompatible SHAP is explicitly recorded, never silently
represented as a completed explanation. Neural diagnostics use feature
permutation and temporal order shuffling. Correlated derived features can make
permutation unrealistic; use the retrained schema ablations as stronger evidence.
No KernelSHAP over every window is launched.

## Optional isolated shifts

```bash
python run_transfer_study.py shifts --data-root out --out out_transfer_final/primary \
  --shift-runs 30 --shift-suites drift strength bursty visibility distance
```

This explicitly generates only compact target evaluation sufficient statistics,
using existing simulator functions, with separate seeds and saved conditions.
No raw photon records are saved. Frozen source checkpoints are evaluated using
their original preprocessing and threshold. If the generating simulator differs
from historical training data, the command refuses unless you explicitly add
`--allow-new-simulator`; that comparison is then also a simulator-version shift.
Weak-strength/distance suites may overlap the training support, so they are
called distribution changes, not automatically unseen-parameter generalization.
E91 distance is not invented where the existing simulator does not support it.

## Scientific claims and limitations

The new code does not make existing physical approximations disappear. E91
coincidence-filtered CHSH is not device-independent security certification.
BKM07 optical results depend on whether the original generation enabled the
extended model. An allowed honest channel and attack with identical observation
distributions cannot be distinguished by these features. No classifier score
replaces authentication, reconciliation, privacy amplification or finite-key
security analysis. Cross-protocol transfer and representation usefulness remain
hypotheses until the new experiments are run.

The reusable code is an empirical research tool, not a publication-ready result.
Recent temporal-QBER/SHAP and normal-operation anomaly-detection QKD work mean
that neither adding an encoder nor adding SHAP establishes novelty by itself.
