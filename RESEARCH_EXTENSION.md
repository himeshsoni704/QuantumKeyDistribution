# Research experiment extension

This implements an experimental JEPA-inspired telemetry encoder, additional independent calibration runs, isolated distribution shifts, and optional detector extensions. No new accuracy or novelty claim is established by adding these modules.

## Run

Install the repository dependencies including PyTorch. Use a new output directory:

```bash
python qkd_pipeline.py selftest
bash run_research.sh core out_research
bash run_research.sh shifts out_research
bash run_research.sh transfer out_research
```

Each stage is separate so expensive transfer runs are optional. Core compares classical baselines, no-AE mean pooling, AE LSTM, JEPA LSTM and JEPA Transformer. Defaults retain five model seeds, 100 maximum epochs per training phase, and early stopping after 10 stale validation epochs. Generation, especially BKM07, can be costly; no completion-time guarantee is possible. Do not rerun `core` into an existing generated directory. Individual train/calibrate/evaluate commands permit continuing after generation.

## Calibration and shifts

There are 185 additional independently seeded runs per class in each calibration partition. Together with the original 15, this gives 200 per class in cal_A and cal_B, without reducing train/test support. This improves empirical precision but does not guarantee deployment FAR, particularly under shift. The extra data are never admitted to the normalizer or training set. Held-out families remain excluded from calibration.

The isolated audit loads fixed checkpoints, normalizers and thresholds; it does not tune on its new data. It varies drift, attack strength, burst noise, E91 visibility, and BB84/BKM07 distance separately. Visibility is only tested for E91; distance is not claimed for E91. Distance shifts move the sampled interval up by 25 km (BB84) or 5 km (BKM07), so they include both overlap and an extrapolation region. Report these as interval shifts, not exclusively unseen distances. Use recorded metadata to report the extrapolation subset separately.

## Optical extensions

`--hardware-effects` enables BB84 afterpulsing (0.01) and two-slot dead time, using the existing implementation. E91 gains independent threshold detectors on both arms with efficiency 0.2 and dark-event probability 1e-5 per emitted-pair slot. A dark-only click produces an independent random outcome. Zero outcomes represent no detection; key and CHSH statistics use coincidences. Exact K now counts detected key-setting coincidences. Monitoring also receives observed coincidence rate. These numbers define a sensitivity study, not calibrated apparatus specifications.

E91 assumes independent, outcome-independent arm losses and coincidence postselection. This does not close the detection loophole or establish device-independent security. BKM07 retains its previous two-leg efficiency abstraction and no dark-count model; a validated apparatus-specific BKM07 optical model remains future work. The existing asymptotic rate features are not operational security guarantees.

## JEPA-inspired representation

Use `--feature-set jepa --model-id jepa_lstm`. The adapter and window encoder produce latent vectors. During honest-only pretraining, 25% of windows are hidden from a GRU predictor. The predictor estimates teacher latent vectors at those positions; the teacher is an exponential moving average (0.99) of the online encoder. The objective combines latent prediction MSE and a variance penalty against collapse. Validation uses a deterministic mask. The supervised MLP consumes original measurements, masks, counts and learned latent features. It does not receive a reconstruction anomaly score from an untrained decoder. AE anomaly detection remains a separate comparison.

This is inspired by Assran et al., CVPR 2023, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, https://arxiv.org/abs/2301.08243. It is a telemetry adaptation with an additional variance penalty, not a reproduction of I-JEPA or a demonstration of semantic understanding. Monitor representation variance, prediction loss, and downstream ablations before making a useful-representation claim.

## Research hypothesis

Test whether predicting temporal measurement relationships transfers to another QKD protocol with fewer target training labels while maintaining detection at a fixed honest false-alert rate under benign channel shifts. Compare JEPA, honest AE and no-pretraining scratch with identical target examples, validation and calibration support. Report all source/target directions, negative transfer and target data use. Cross-protocol transfer is a research question; an adapter or encoder alone is not proof of novelty. A broader prior-work review is still necessary before a first-in-literature claim.

The implementation is for offline complete-run inference. Window boundaries depend on the completed exact-K session; prefix metrics must not be presented as a deployment-ready real-time monitor.
