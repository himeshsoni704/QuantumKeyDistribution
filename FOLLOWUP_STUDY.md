# Baseline audit and independent-data confirmation

This follow-up reads the existing `out_full_new` and `out_transfer_final` outputs,
without rewriting them or changing the legacy simulator/transfer runner. The
entry point is `tools/qkd_followup.py`. The default new output is
`out_followup_20261121`, excluded from Git along with generated datasets/logs.

## What runs

1. **Saved controls:** inspect all five saved study stages, not just the four
   models in the original review. Export seed-level metrics and per-attack
   rates for all available test evaluations; create BB84 feature/control heatmaps.
   A coverage file lists missing prediction files. A missing comparison is never
   interpreted as zero error. Fitted code hashes, dataset hashes and prediction
   group/label identities must match before results are combined.
2. **Threshold diagnosis:** retain each strict source model, normalizer and
   temperature; score target `cal_B` honest sessions only to choose a separate
   target threshold. Compare its test recall/FAR with the original threshold.
   Target test labels never choose a threshold. This is explicitly **target-honest
   adaptation**, not zero-shot. Test AUROC describes discrimination independent
   of the alarm cut-off. Reference/test/source-fitting overlap is rejected.
3. **Historical targeted repeat:** BKM07 to E91, five training runs per original
   target class, JEPA/AE/MLP head-only versus their matched scratch fits. Also fit
   logistic, RF and QBER controls on the same selected target groups, with the
   same validation/calibration exposure. Preserve paired group bootstrap outputs,
   label usage and both absolute error rates. The historical repeat remains
   exploratory; it is not new independent evidence.
4. **Fresh replications:** repeat on three independently simulated datasets with
   data seeds 2026112101, 2026112102, 2026112103, using the unchanged simulator,
   original key-event budget and operating distributions. Each protocol has:

   | Split | Sessions |
   |---|---:|
   | Train | 210 per original class |
   | Validation | 30 per original class |
   | Temperature calibration (`cal_A`) | 50 per original class |
   | Honest threshold calibration (`cal_B`) | 200 honest only |
   | Test | 300 honest, 100 per attack family |

   These defaults require **17,640 new simulated sessions** in total. No raw
   photon arrays are persisted. Generation is sequential to bound peak memory,
   saves sufficient statistics after every session and resumes completed sessions.
   Exact-K simulation can be substantial CPU work; no unsupported ETA is given.
   Fresh BB84 controls compare QBER, common, common-with-support, full and specific
   features, with logistic, RF, MLP, AE and JEPA models; AE/JEPA use latent and
   combined representations. Five training seeds reuse each dataset's split.
   Epoch cap is 100 with early stopping, not a mandatory 100 epochs.

The **preselected primary replication contrast** is JEPA head-only minus JEPA
scratch macro-F1 for BKM07 -> E91 at budget 5/class. Absolute recall/FAR, simple
controls and BB84 feature ablations are also reported. Do not select a new winner
from the test results and call it the preselected hypothesis.

`replication_dataset_means.csv` averages training seeds within each fresh dataset.
`replication_summary.csv` uses data-generation seed as the replication unit.
Three datasets are limited replication evidence, not a strong population-level
significance result. Other feature/model comparisons remain exploratory.
A fixed 5% calibration setting is not a guarantee on a shifted target.

## Plan, launch and monitor

Use the environment that already ran the study (default `.venv/bin/python`):

```bash
cd /home/himesh/Documents/QuantumKeyDistribution
bash run_followup.sh plan
bash install_followup_service.sh
```

The installer enables and starts a **user systemd service**, so it survives closing
its launching terminal. It needs the normal desktop user service manager. It does
not request root privileges or change machine-wide login/linger settings. A user
service's lifetime after full logout depends on the host's existing user-manager
policy; no persistence across full logout/power loss is promised. It starts on a
future user-manager start and safely reuses completed jobs. Code/config changes
require a new output directory.

```bash
# Live log (includes generation counts, training and validation progress)
tail -f /home/himesh/Documents/QuantumKeyDistribution/logs/qkd_followup.log

# Service state
systemctl --user status qkd-followup.service --no-pager

# Current machine-readable stage and latest counts
cat /home/himesh/Documents/QuantumKeyDistribution/out_followup_20261121/status.json

# Stop / resume after interruption
systemctl --user stop qkd-followup.service
systemctl --user start qkd-followup.service
```

`status.json` is an atomic snapshot and `events.jsonl` retains stage transitions.
The service stops on failure, records the error and leaves caches available for
resumption. It does not repeatedly restart a failed simulation. A lock rejects a
second launcher for the same output directory. GPU use is detected in the actual
service environment; CPU is supported. The generated service uses the repository
under `~/Documents/QuantumKeyDistribution`; edit its paths for a different checkout.

For a bounded development check use another output directory and explicitly
reduced counts/epochs. Such a check validates execution, not scientific accuracy:

```bash
QKD_FOLLOWUP_OUT="$PWD/out_followup_development" bash run_followup.sh all \
  --data-seeds 2026112199 --model-seeds 0 --epochs 2 --bootstrap 10 \
  --train-per-class 10 --val-per-class 5 --calibration-per-class 5 \
  --honest-calibration 20 --honest-test 20 --attack-test 5
```

## Outputs

- `followup_contract.json`: code and configuration lock.
- `saved_review/`: full control tables, BB84 heatmaps, coverage and threshold plots.
- `historical_confirmation/`: matched historical fits, predictions and paired gains.
- `fresh_data_<seed>/`: new data only, split metadata, per-session caches, hashes.
- `replication_<seed>/`: new confirmation and BB84 controls on each dataset.
- `replication_all_gains.csv`, `replication_dataset_means.csv`,
  `replication_summary.csv`: accumulated independent-data replication summaries.

Raw scores, thresholds, original attack labels and test group IDs are retained in
prediction files. Rounded charts are descriptive summaries, not additional test
observations. QKD security certification and real-time detection are outside scope.

## Focused BKM07 to E91 replication

The focused mode generates BKM07 and E91 datasets and runs the matched
five-per-class experiment. It uses the historical BB84 configuration JSON
to reproduce the simulator settings. It skips historical fitting, saved-output
review, and BB84 generation and controls. Use fresh data seeds for a further
independent replication. The example below generates 10,590 sessions across
three datasets. Each dataset has 300 honest E91 test sessions and 100 per attack
family.

```bash
project=/home/himesh/Documents/QuantumKeyDistribution
study="$project/out_bkm07_e91_confirmation_20261124"
cd "$project" || exit 1
git fetch origin codex/bkm07-e91-confirmation
git worktree add --detach "$project/../qkd-bkm07-e91-confirmation" FETCH_HEAD
cd "$project/../qkd-bkm07-e91-confirmation" || exit 1

"$project/.venv/bin/python" tools/qkd_followup.py plan --focused \
  --data-seeds 2026112401 2026112402 2026112403
mkdir -p "$study/matplotlib"
export MPLCONFIGDIR="$study/matplotlib"
flock -n "$study/.run.lock" "$project/.venv/bin/python" -u tools/qkd_followup.py replicate \
  --focused --data-root "$project/out_full_new" \
  --existing-study "$project/out_transfer_final" --out "$study" \
  --data-seeds 2026112401 2026112402 2026112403 2>&1 | tee -a "$study/run.log"
```

Monitor `run.log` and read `status.json` in the study directory. Repeat the same
training command with the same checkout and options to resume an interrupted
run. A different code revision or experiment configuration requires a fresh
output directory.

The study saves scratch and transferred absolute metrics, paired test-session
intervals for each training seed, per-attack results, and frozen-source versus
honest-target threshold metrics for MLP, AE, and JEPA. Logistic regression, RF,
and QBER are target-trained controls. `replication_paired_intervals.csv`
resamples dataset-level means of paired gains. It omits interval bounds until
three datasets have completed. Three independent datasets still provide limited
evidence.
