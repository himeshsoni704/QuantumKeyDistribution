# Research experiment extension

This implements an experimental JEPA-inspired telemetry encoder, additional independent calibration runs, isolated distribution shifts, and optional detector extensions. No new accuracy or novelty claim is established by adding these modules.

## Run

Install the repository dependencies including PyTorch. Use a new output directory:

```bash
python qkd_pipeline.py selftest
bash run_research.sh core out_research
bash run_research.sh shifts out_research
bash run_research.sh transfer out_research
# Optional: all saved transfer methods under each isolated shift (many evaluations)
bash run_research.sh transfer-shifts out_research
```

Each stage is separate so expensive transfer runs are optional. Core compares classical baselines, no-AE mean pooling, AE LSTM, JEPA LSTM and JEPA Transformer. Defaults retain five model seeds, 100 maximum epochs per training phase, and early stopping after 10 stale validation epochs. Generation, especially BKM07, can be costly; no completion-time guarantee is possible. Do not rerun `core` into an existing generated directory. Individual train/calibrate/evaluate commands permit continuing after generation.

## Calibration and shifts

There are 185 additional independently seeded runs per class in each calibration partition. Together with the original 15, this gives 200 per class in cal_A and cal_B, without reducing train/test support. This improves empirical precision but does not guarantee deployment FAR, particularly under shift. The extra data are never admitted to the normalizer or training set. Held-out families remain excluded from calibration.

The isolated audit loads fixed checkpoints, normalizers and thresholds; it does not tune on its new data. It varies drift, attack strength, burst noise, E91 visibility, and BB84/BKM07 distance separately. Visibility is only tested for E91; distance is not claimed for E91. Distance shifts move the sampled interval up by 25 km (BB84) or 5 km (BKM07), so they include both overlap and an extrapolation region. Report these as interval shifts, not exclusively unseen distances. Use recorded metadata to report the extrapolation subset separately.

## Optical extensions

`--hardware-effects` enables BB84 afterpulsing (0.01) and two-slot dead time, using the existing implementation. E91 gains independent threshold detectors on both arms with efficiency 0.2 and dark-event probability 1e-5 per emitted-pair slot. A dark-only click produces an independent random outcome. Zero outcomes represent no detection; key and CHSH statistics use coincidences. Exact K now counts detected key-setting coincidences. Monitoring also receives observed coincidence rate. These numbers define a sensitivity study, not calibrated apparatus specifications.

E91 assumes independent, outcome-independent arm losses and coincidence postselection. This does not close the detection loophole or establish device-independent security. BKM07 now has an optional explicit two-way optical model, described below. Apparatus validation remains necessary. The existing asymptotic rate features are not operational security guarantees.

## JEPA-inspired representation

Use `--feature-set jepa --model-id jepa_lstm`. The adapter and window encoder produce latent vectors. During honest-only pretraining, 25% of windows are hidden from a GRU predictor. The predictor estimates teacher latent vectors at those positions; the teacher is an exponential moving average (0.99) of the online encoder. The objective combines latent prediction MSE and a variance penalty against collapse. Validation uses a deterministic mask. The supervised MLP consumes original measurements, masks, counts and learned latent features. It does not receive a reconstruction anomaly score from an untrained decoder. AE anomaly detection remains a separate comparison.

This is inspired by Assran et al., CVPR 2023, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, https://arxiv.org/abs/2301.08243. It is a telemetry adaptation with an additional variance penalty, not a reproduction of I-JEPA or a demonstration of semantic understanding. Monitor representation variance, prediction loss, and downstream ablations before making a useful-representation claim.

## Research hypothesis

Test whether predicting temporal measurement relationships transfers to another QKD protocol with fewer target training labels while maintaining detection at a fixed honest false-alert rate under benign channel shifts. Compare JEPA, honest AE and no-pretraining scratch with identical target examples, validation and calibration support. Report all source/target directions, negative transfer and target data use. Cross-protocol transfer is a research question; an adapter or encoder alone is not proof of novelty. A broader prior-work review is still necessary before a first-in-literature claim.

The implementation is for offline complete-run inference. Window boundaries depend on the completed exact-K session; prefix metrics must not be presented as a deployment-ready real-time monitor.

## Explicit BKM07 optical model

With `--hardware-effects`, the travelling photon survives forward fibre with
probability t = 10^(-alpha*d/10). Bob chooses SIFT or CTRL independently of
arrival. CTRL reflection has efficiency r and does not require a Bob detector
click. SIFT uses a Bob threshold detector of efficiency eta_B and background
probability d_B. Bob resends only after a click, with source efficiency s. Alice
receives through the return fibre and measures with efficiency eta_A and
background probability d_A. Dark-only detections have random bits. A Bob
dark-only click can trigger preparation; it does not recreate Alice's original
bit. Key opportunities require both a Bob SIFT bit and an Alice click.

With no dark counts/dead time, conditional signal detection probabilities are
P(A click | CTRL) = t^2*r*eta_A and
P(A and B clicks | SIFT) = t^2*eta_B*s*eta_A.
The new test checks that CTRL bypasses Bob's detector, whereas SIFT includes it.
Both detectors support dead slots and next-slot afterpulsing. Detector history
restarts at continuation batch boundaries; the default enables dark counts but
leaves these memory effects disabled for BKM07. These are explicit simulation
assumptions, not a complete model of a particular laboratory system.

## Encoder alternatives and combinations

| Variant | Honest pretraining | MLP inputs | Reconstruction anomaly output |
|---|---|---|---|
| no-AE | None | Measurements, masks, counts | No |
| AE | Reconstruct observed measurements | Original inputs + latent + residuals + error | Yes |
| JEPA-inspired | Predict withheld latent windows, EMA teacher, variance penalty | Original inputs + latent | No |
| Hybrid | Both objectives on one shared encoder | Original inputs + latent + residuals + error | Yes |

The hybrid uses a shared encoder trained with both objectives. It does not stack
an AE after a JEPA or concatenate two independently trained encoders. The runner
compares LSTM/Transformer versions of JEPA and hybrid, as well as the AE and
simple baseline. No variant is assumed superior in advance.

Transfer now runs no-AE, AE, JEPA and hybrid separately, with representation names
in checkpoint and output filenames to prevent overwriting. Each target uses
5, 20, 100 or all training labels per class; scratch, head-only and partial
fine-tuning get identical sampled target groups. Reports include recall change,
false-alert change, actual honest pretraining support, validation/calibration
label costs, and exact binomial intervals for false alerts and recall. Meeting
the empirical false-alert target is reported separately from uncertainty; it is
not a deployment guarantee. `transfer-shifts` evaluates all saved target transfer
checkpoints using the original calibration thresholds without retraining.

A coincidence-only E91 Bell violation assumes fair sampling of detected pairs.
The simulator retains non-detections but its key/correlation estimators select
coincidences; it therefore does not prove device-independent security. Such a
claim would need an applicable loophole-aware security proof, finite-key
analysis and classical postprocessing. See Orsucci et al., Quantum 4, 238
(2020), https://doi.org/10.22331/q-2020-03-02-238 and the 2022 DIQKD experiments.
