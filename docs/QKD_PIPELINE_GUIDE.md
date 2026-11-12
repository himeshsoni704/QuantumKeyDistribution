# QKD pipeline guide
**BB84 · BKM07 · E91 | From simulated photons to transfer results**

This guide describes the simulator in `qkd_pipeline.py` and the binary transfer study launched by `run_full.sh` on `codex/standalone-qkd-dl`.

## 1. Follow one run
Imagine Alice and Bob repeatedly exchange quantum signals, then compare a sample of their results. One simulation run records this whole session. The machine-learning model sees statistics from the session and scores whether it looks like one of the simulated attack scenarios.

1. **Simulate.** Choose a protocol, a channel and an honest or attacked scenario. Draw random signals, losses, clicks and outcomes.
2. **Sift.** Keep eligible outcomes according to that protocol. Use comparisons as disclosed test information.
3. **Count.** Split the emitted rounds into 50 windows by default. Keep numerators and denominators for measurements in each window.
4. **Split.** Assign complete runs to training, validation, calibration or test. Keep stress runs separate.
5. **Fit.** Train a baseline or a small neural model and choose its checkpoint using validation.
6. **Calibrate.** Turn scores into an alarm decision using dedicated calibration runs.
7. **Evaluate.** Compare predictions with held-out run labels, including other protocols.

The default simulator collects 2,000 key-contributing events per run. A lossy session can require more emitted rounds to reach this count. A window is a fraction of the completed session, so this is an offline monitoring study.

### The two entry points
- `python qkd_pipeline.py gen --out …` creates a dataset. Generation writes feature arrays and raw records, which can be large.
- `bash run_full.sh INPUT OUTPUT run` reads existing feature arrays and trains the new study. It writes its own checkpoints and results.

<!-- page -->

## 2. What the simulator does
**BB84.** Alice picks bits, bases and pulse intensities. Bob picks measurement bases. Matching bases on detected signal rounds provide the key comparisons. Weak decoys and vacuum slots help estimate intensity-dependent detection rates. The attack menu includes intercept-resend, photon-number splitting, blocking and loss manipulation.

**BKM07.** Alice sends a quantum signal to Bob and receives one back. In CTRL Bob reflects it. In SIFT he measures in the computational basis and sends a prepared result. Alice checks the appropriate returned outcome against the relevant preparation or Bob's recorded bit. The simulator compares symmetric and asymmetric interception across the two legs. The optional optical mode distinguishes the forward and return transmission paths, Bob's SIFT detection and Alice's return detection.

**E91.** Alice and Bob each measure one half of a simulated entangled pair using chosen angles. In ideal same-angle singlet measurements their outcomes are opposite. Bob's sign is flipped for key comparison. The simulator also measures correlations across settings and computes a CHSH statistic. Its scenarios include intercept-resend, a simplified ancilla model and extra depolarization.

### Channel noise, in everyday terms
- **Fibre loss** reduces the fraction of signals that arrive. With attenuation a in dB/km and distance L in km, transmission is 10^(-aL/10). Independent losses on two legs multiply when both must succeed.
- **Weak coherent pulses** have a random photon count. For mean intensity mu, P(N = n) = exp(-mu) mu^n / n!. A vacuum pulse has mu = 0.
- **Detector efficiency** gives a surviving photon another chance to be missed. **Dark counts** can create clicks without a signal. **Afterpulsing** can create a later extra click. **Dead time** suppresses nearby detections. The available effects depend on the protocol and generation settings.
- **Drift and bursts** vary the channel during a session. Drift is a smooth sinusoidal change and bursts are localized changes. These are simulator profiles.
- **E91 visibility** controls how strong the ideal singlet correlations look. For ideal polarization settings with visibility V, the correlation is -V cos(2(a-b)). The optical mode can discard rounds without a coincidence.

Honest noise and an attack can produce overlapping measurements. A classifier can only distinguish differences visible in its inputs. Coincidence-filtered CHSH and the classifier score do not establish device-independent key security.

<!-- page -->

## 3. From clicks to model inputs
Think of each run as a spreadsheet with 50 rows. Each row describes one window. For each measurement the dataset stores what happened and how many opportunities there were.

**QBER example.** If 3 of 40 eligible comparisons disagree, QBER = 3 / 40 = 7.5%. If there are no comparisons, that window's QBER is unavailable. The model receives a mask to distinguish missing values from a genuine zero.

**Gain example.** If 8 of 200 weak-decoy slots click, weak-decoy gain = 8 / 200 = 4%. BB84 tracks signal, weak-decoy and vacuum gains separately.

**Correlation example.** For E91, multiply Alice's and Bob's outcomes on the selected setting pair, then average those products. CHSH combines four setting correlations with the simulator's sign convention. CHSH and QBER answer different physical questions.

### What features travel between protocols?
- **QBER** uses key-disagreement rates. The threshold baseline averages available window QBERs.
- **Common** adds the change from the preceding window and its absolute value. Both adjacent windows need valid observations.
- **Common support** adds log(1 + number of disclosed comparisons). This tests whether sample size carries helpful information or makes the protocol easier to recognize.
- **Full** keeps the observable feature union, excluding the duplicate BKM07 error field. **Specific** additionally drops the common key-disagreement rate. These views retain information such as decoy gains and E91 correlations. Strict zero-shot and leave-one-protocol-out use the portable views.

### Who gets to see each split?
Training fits model weights and any scaler. Validation chooses the neural checkpoint. Calibration A fits a score temperature where applicable. Honest runs in calibration B set the alert threshold. Test groups are scored after these choices are fixed. The runner checks that a simulation group cannot cross splits.

A raw input uses its original scale. Standardization subtracts the source-training mean and divides by the source-training spread. The honest-reference version uses honest observations to estimate that scale. Using a target honest reference is its own adaptation experiment.

<!-- page -->

## 4. Models and transfer experiments
**Classical baselines.** Logistic regression learns a weighted sum of run summaries. Random Forest votes across many trees. Boosting builds trees that improve on earlier predictions. These models use summaries such as mean, spread and availability.

**MLP.** A small network reads each window and pools the window representations into a run score.

**AE + LSTM.** The autoencoder compresses honest measurements into a small vector and learns to reconstruct them. The encoder's vector becomes a learned feature. An LSTM reads the windows in order and its outputs are pooled for the run decision.

**JEPA-inspired + LSTM.** Some windows are hidden during honest pretraining. A predictor uses visible context to predict the hidden windows' embeddings from a slowly updated teacher. The downstream classifier uses the learned encoder. The pretraining predictor is discarded. The latent-only and combined experiments test learned features alone and alongside original measurements.

### Five ways to ask the research question
1. **Within protocol.** Train and test on separate runs from the same protocol.
2. **Strict zero-shot.** Train on one protocol and score a different one using the same weights, input mapping, scaler and threshold. All six ordered pairs are evaluated.
3. **Few-shot.** Use 5, 20, 100 or all target training runs per original simulator class. Compare scratch, a new head on frozen source features, and partial tuning of the temporal readout where present. Target validation and calibration are counted separately. Budget zero uses strict zero-shot.
4. **Leave one protocol out.** Train on two protocols and score the third with source-only fitting and calibration.
5. **Joint.** Train a shared binary model with data from all three and evaluate held-out runs for each. Every test protocol was represented in training.

The optional target honest-reference comparison measures what changes when a separate honest target sample supplies the input normalization. It retains the source model and threshold.

<!-- page -->

## 5. How to read the results
Start with `out_transfer_final/primary/results.csv`. Each row identifies its experiment, model, source, target, split and seed. Each stage has a `summary.csv` with across-seed summaries.

- **Attack recall** is the fraction of attacked runs flagged at the chosen threshold.
- **False-alarm rate** is the fraction of honest runs flagged. Check it alongside recall.
- **Macro-F1** gives the honest and attack classes equal weight when averaging their individual F1 scores.
- **AUROC** measures how well the scores rank attacked runs above honest ones. **Average precision** summarizes precision versus recall and depends on attack prevalence.
- **Brier score and reliability bins** show how scores behave as probability estimates.
- **Confidence intervals** resample complete runs. Paired gains use the same held-out target runs for transferred and scratch models. The five training seeds use the saved dataset split.

A frozen source threshold can produce many target false alarms even when target AUROC looks promising. With too few honest calibration runs, the conservative threshold can suppress all alerts. Look for `insufficient_resolution` in the calibration information.

### Where to look next
- **primary** contains the core comparisons. **qber** gives the simple reference.
- **representations** compares scaling and raw/learned feature choices.
- **full_features** tests richer within-protocol and joint inputs. **support** adds comparison counts.
- `transfer_gaps.csv`, `paired_gains.csv` and `label_efficiency.csv` help explain whether transfer helps and how it changes with target training labels.
- Diagnostics test whether embeddings encode attack status or protocol identity. Permutation and window shuffling measure model reliance. Optional TreeSHAP describes tree-model scores. These are diagnostics of model behavior.
- `predictions.npz`, checkpoints, training history and `study.json` record the evidence behind summaries.

Start by comparing within-protocol and strict zero-shot rows on recall and false alarms. Then check paired few-shot gains and whether learned features improve on the simple QBER and classical baselines.

<!-- page -->

## 6. Pull, choose inputs and clean up
Use the Python environment that successfully ran the earlier project. Activate it before launching the job. PyTorch uses CUDA when available, while tree fitting and other stages also use CPU.

```bash
cd ~/Documents/QuantumKeyDistribution
git status --short
git switch codex/standalone-qkd-dl
git pull --ff-only origin codex/standalone-qkd-dl
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If Git stops because of local changes, review and save those changes before retrying. The example assumes your old input is `out`. Substitute your actual directory, such as `qkd_out_full`.

```bash
DATA_ROOT="$PWD/out"
RESULTS="$PWD/out_transfer_final"
ls "$DATA_ROOT/data/feature_schema.json"
du -sh "$DATA_ROOT" "$RESULTS" 2>/dev/null
df -h .
```

### Start clean or resume
For a clean rerun, first verify no prior run is writing the results. Archive only the previous **transfer results**. The input directory contains the generated data needed for reuse. Leave the existing results folder in place if you intend to resume.

```bash
if [ -d "$RESULTS" ]; then
  mv -- "$RESULTS" "${RESULTS}.old.$(date +%Y%m%d-%H%M%S)"
fi
ls -ld -- "${RESULTS}".old.* 2>/dev/null
```

If you need the space, review the listing and delete the specific archived results folder interactively with `rm -rI -- /exact/path/to/archive`. This deletion is permanent. Keep the data directory, particularly its `feature_schema.json` and each protocol's arrays and split metadata. No cleanup is required for a normal run with a new results name.

### If you deliberately want new simulated data
Generate in a new directory, then point `DATA_ROOT` to it. Enable extended optics explicitly if desired. This step can take substantial time and storage.

```bash
python -u qkd_pipeline.py gen --out out_generated_v2 --hardware-effects
DATA_ROOT="$PWD/out_generated_v2"
```

<!-- page -->

## 7. Launch with nohup and monitor
From the repository folder, with `DATA_ROOT` and `RESULTS` set as on the previous page, print the experiment count and storage estimate first.

```bash
bash run_full.sh "$DATA_ROOT" "$RESULTS" plan
```

Launch the study in the background. Keep the log and PID file outside the results directory. The wrapper writes the exit status to the log when the run ends.

```bash
LOG="$PWD/transfer_run_$(date +%Y%m%d-%H%M%S).log"
nohup bash -c '
  bash run_full.sh "$1" "$2" run
  rc=$?
  echo "Transfer study exit status: $rc"
  exit "$rc"
' _ "$DATA_ROOT" "$RESULTS" > "$LOG" 2>&1 < /dev/null &
echo "$!" > transfer_run.pid
printf '%s\n' "$LOG" > transfer_run.logpath
tail -f "$LOG"
```

Press Ctrl+C to leave `tail`. The background job continues. `nohup` survives a normal terminal disconnect on the same machine. A machine shutdown or hosted runtime expiry can still interrupt it.

After reconnecting, open the repository folder and check:

```bash
ps -p "$(cat transfer_run.pid)" -o pid,etime,stat,cmd
tail -n 40 "$(cat transfer_run.logpath)"
nvidia-smi
```

Exit status 0 means the wrapper finished successfully. A nonzero status means inspect the end of the log. A missing status line can mean the wrapper is still running or was interrupted. Before restarting, check the old job and its children. Run only one writer per results directory.

To resume, repeat the launch with the same input, results directory, code and settings. Completed jobs are reused. An interrupted fit can restart. To add the optional compact shift experiments after training:

```bash
python -u run_transfer_study.py shifts --data-root "$DATA_ROOT" \
  --out "$RESULTS/primary" --shift-runs 30
```

Historical data and a newer shift simulator can differ. The shift runner checks that provenance and requires explicit opt-in for a cross-version comparison.
