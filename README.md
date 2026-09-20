# QKD Eavesdropping Detection with Machine Learning

Simulates three quantum-key-distribution protocols — **BB84**, **BKM07**
(semi-quantum), and **E91** (entanglement-based) — under several
eavesdropping strategies, extracts a physically-grounded feature vector
from each simulated run, and trains classifiers to distinguish a secure
channel from an attacked one. The centerpiece is not the classifiers
themselves (six fairly standard scikit-learn models) but the discipline
around them: a channel model anchored to published experimental
parameters, an eight-test leakage audit that runs *before* any model is
trusted, an ablation study that attributes detection performance to
specific feature groups, and four generalisation tests that probe whether
a model learned an attack signature or just memorised a distribution.

This document explains the working methodology — what's simulated, why
each design choice was made, and how the pieces fit together. For exact
formulas and line-level detail, read the notebook itself
(`qkd_ml_v2.ipynb`); every code cell has an accompanying markdown cell
explaining the physics or the feature it implements.

---

## 1. Why three protocols, and what's different about each

| Protocol | Nature | What Eve has to attack |
|---|---|---|
| **BB84** (Bennett & Brassard, 1984) | Fully quantum | Alice sends single photons (weak coherent pulses) in one of two bases; a single forward leg |
| **BKM07** (Boyer, Kenigsberg & Mor, 2007) | Semi-quantum | Bob is classical (measure-in-Z or reflect); the photon makes a **round trip**, so both the forward and return leg are attack surfaces |
| **E91** (Ekert, 1991) | Entanglement-based | A shared source distributes entangled pairs; security rests on a genuine Bell-inequality (CHSH) violation, not on Eve disturbing a *sent* state |

Because the three protocols have structurally different security
mechanisms, they need different feature sets and different honest-noise
models — there is no single "one noise model fits all" shortcut without
losing exactly the physics that makes each protocol's detection problem
interesting (see Section 6, the matched-QBER calibration layer, for how
they're nonetheless compared fairly).

## 2. Channel physics: what's real and what's a knob

### BB84 / BKM07 — GYS fibre-link model

Honest-channel noise is **not** a free parameter. `channel_model()`
implements the closed-form gain/QBER equations of Ma, Qi, Zhao & Lo
(*Phys. Rev. A* 72, 012326, 2005), Eqs. (4)-(11), with default constants
calibrated to the Gobby–Yuan–Shields experiment (*Appl. Phys. Lett.* 84,
3762, 2004):

| Constant | Meaning | GYS value |
|---|---|---|
| `alpha_db_km` | Fibre attenuation @ 1550nm | 0.21 dB/km |
| `eta_bob` | Bob-side transmittance × detector efficiency (combined, not raw detector efficiency alone) | 0.045 |
| `Y0` | Background/dark yield per pulse | 1.7×10⁻⁶ |
| `e_detector` | Optical misalignment error rate (distance-independent floor) | 0.033 |
| `e_0` | Error rate of a background/dark click | 0.5 (definitional — a dark click carries no information) |

This is what makes the honest QBER **flat (~3.3%) at short range and only
rise toward 50% at very long range**, matching the literature, rather than
the naive mistake of adding a fixed `p_jitter` on top of a raw noise
formula (which nearly doubles QBER over 0-15km and saturates by 50km —
physically wrong; an early defect in this project, since corrected).

BKM07 reuses the same `channel_model()`, applied **twice** (forward +
return leg: `simulate_bkm07_pulse`), so round-trip survival is
`eta_1way**2`. A subtlety worth stating plainly, because it isn't obvious
from the code at a glance: in the current model, **`distance_km` does not
change BKM07's honest per-round error rate at all** — it only changes
round-trip *survival* probability. The error rate is driven entirely by
`e_detector`, composed across three independent noisy operations per
SIFT_KEY round trip (Bob's measurement, Bob's re-preparation, Alice's
final measurement) via the standard binary-symmetric-channel composition
identity. This is exactly the kind of implicit assumption the matched-QBER
calibration layer (Section 6) makes explicit and calibratable, rather than
leaving it as a hidden fact about the simulator.

### E91 — genuine two-qubit density matrix

The honest channel is a **Werner state**, `rho_W = V|Psi-><Psi-| + (1-V)
I/4`, giving `|S|_max = 2*sqrt(2)*V` and key-basis QBER `= (1-V)/2` exactly
(Werner, *Phys. Rev. A* 40, 4277, 1989). Every attack is implemented as a
genuine **CPTP map** on that state (measure-and-resend, an entangling
ancilla probe, asymmetric arm loss) and outcomes are sampled from the
exact Born-rule joint distribution — not looked up from a precomputed
correlation table. This matters: an earlier table-lookup version of this
simulator had a basis-convention bug that made 100% intercept-resend
measure `|S| = 2.77`, above the Tsirelson bound and therefore physically
impossible; the density-matrix simulation cannot make that mistake by
construction.

## 3. Feature engineering, per protocol

Each protocol compresses one simulated run into a fixed-length feature
vector spanning three information sources: **aggregate error rates**
(what a conventional threshold detector already uses), **decoy-state /
photon-number estimators** (BB84 only — the only way to see a
Photon-Number-Splitting attack, since Bob cannot count photons directly),
and **temporal structure** (variance, spectral entropy, autocorrelation of
the per-window error trace — the only way to see an attack whose *average*
QBER contribution is small but whose *timing* is non-stationary).

The full, current feature list for each protocol — with the reasoning
behind every entry — lives in the notebook's Section 3 (BB84/BKM07) and
Section 3b (E91) markdown cells; it's intentionally not duplicated in
full here; treat those cells as the authoritative reference, and this
README as the map to them.

One feature-extraction gap found and fixed during this work: BKM07's
`collect_bkm07_features()` computed `qber_zs` from a variable that gets
reassigned mid-function (after a re-preparation flip), making it a
2-flip composite rather than the true end-to-end round-trip key error a
real deployment would report. A new `qber_key` field (`bit_A != bit_A_final`
over SIFT_KEY rounds) was added as BKM07's analogue of BB84's
`qber_total` / E91's `qber_key` — this is the field the matched-QBER
calibration layer targets.

## 4. The Monte Carlo simulators

BB84's `simulate_bb84_decoy` is **vectorised**: every random draw (photon
number, bit, basis, click, error) for all N pulses happens as NumPy array
operations in one call, not a Python loop. That's what makes
`N=2,000,000` pulses/run — the scale needed before the windowed temporal
features rise above their own binomial estimator noise floor — run in
seconds rather than minutes. An earlier, disconnected "large-scale Monte
Carlo" demo cell used a separate, simplified simulator (`simulate_bb84_batch`,
an abstract `noise_prob` model unrelated to the GYS channel); that cell has
been replaced with a direct demonstration of the real simulator, at the
scale feature extraction actually runs at, validated against
`channel_model()`'s closed-form prediction.

BKM07's `simulate_bkm07_pulse` is a genuine per-pulse Python loop — not
vectorised — because round-trip loss means most fired pulses are "lost"
before reaching a measurement, so the useful sample count per call is
naturally much smaller than N, and the loop body is cheap enough that
vectorising it wasn't necessary at the scale this project runs at.

### A reproducibility bug worth naming

All random draws go through a `SeedBook` class that maps a `(role,
index)` pair to an independent, deterministic RNG stream — e.g. every
`(distance, e_detector, Y0)` triple used for run index `i` is shared
across that index's secure and attacked samples (Section 5's "common
random numbers"). `SeedBook` originally derived each stream's root from
Python's built-in `hash(role)`. That looks deterministic within one
notebook run, but `hash()` on a string is salted by `PYTHONHASHSEED`,
which Python randomises **per process** by default since 3.3 (a security
hardening against hash-flooding attacks) — so a fresh kernel or a fresh
`python` invocation gets a *different* tag for the same role string, and
therefore different actual random draws, even though nothing about the
notebook's own code changed. This was caught by a leakage-audit control
(Section 5's L-8) flipping between "OK" and "LABEL LEAKS" across otherwise
identical re-runs — the audit result itself wasn't wrong, the *seeding*
silently wasn't reproducible. Fixed by switching to `zlib.crc32(role.encode())`,
a fixed, unsalted hash, so the same role now maps to the same stream
across separate runs, which is what the class's own docstring always
claimed it did.

A second, more load-bearing instance of the same class of bug was found
right after: `measure_qubit()` — the function BKM07's entire round-trip
simulation calls to decide every measurement outcome — drew its
randomness from the bare `np.random` module instead of the `rng` object
`simulate_bkm07_pulse` carefully threads into every *other* random draw
in that function. `SeedBook`'s deterministic streams were therefore being
correctly constructed and correctly passed around, only to be silently
ignored at the one place that actually determines BKM07's measured bit
values — meaning every BKM07 QBER feature (and the calibration layer's
empirical BKM07 check) was non-reproducible across runs regardless of the
`SeedBook` fix above. Both `measure_qubit()` and the standalone
photon-number demo (`simulate_photon_number_batch`) now take an explicit
`rng` parameter and use it exclusively. **Verified end-to-end**: two
independent full notebook runs after both fixes are diffed line-for-line
identical except for wall-clock timing prints — every QBER measurement,
every hyperparameter-tuning score, every leakage-audit number, and every
model evaluation metric now reproduces exactly.

## 5. Leakage audit — before any model is trusted

Every generated dataset runs through an 8-test audit (Kapoor & Narayanan,
*Patterns* 4, 100804, 2023) before any classifier sees it:

- **L-1** Single-feature AUC (>0.99 flags a feature that's suspiciously
  perfect on its own).
- **L-2/L-3** Exact class separation / per-class constants.
- **L-4** Nuisance-parameter distributions (channel distance, `e_detector`,
  `Y0`) must match across attack classes via a KS test — the single most
  important check, since it's what catches a dataset where the channel
  *realisation itself*, not the attack, differs by class.
- **L-5** Sample counts identical across classes.
- **L-6** Feature NaN-rate parity across classes.
- **L-7** Shuffled-label control: AUC on randomly-shuffled labels must be
  ~0.50.
- **L-8** Zero-strength control: keep the class *labels* but set every
  attack's strength to zero — AUC must collapse to ~0.50, or the label is
  leaking through something other than the simulated attack.

A dataset failing L-4 was a real, found-and-fixed bug in this project:
nuisance parameters (distance, detector error) were originally drawn
independently per class rather than once per run-index and shared across
classes, so a classifier could distinguish classes from the *channel
realisation* alone, with zero attack-detection involved. The fix — draw
nuisance parameters once per run index, before the class loop (`P13`'s
"common random numbers" pattern) — is what makes L-4 pass honestly.

## 6. Matched-QBER calibration layer

`distance_km` (BB84/BKM07) and `V` (E91) are not comparable knobs across
protocols: BB84's honest QBER rises nonlinearly with distance, E91's is
linear in `(1-V)`, and BKM07's turns out to be independent of distance
entirely (Section 2). Sweeping the raw knob and comparing AUC across
protocols would mostly report how each knob happens to map to QBER, not
anything about the protocols' actual detectability. `calibrate_bb84`,
`calibrate_e91`, and `calibrate_bkm07` invert each protocol's own
honest-noise formula (closed-form for all three — BKM07's via the
binary-symmetric-channel composition identity in Section 2) so a future
benchmark can drive all three protocols to the **same target honest
QBER**, and calibrate attack strength to a target *excess* QBER, for an
apples-to-apples cross-protocol comparison. The layer itself is validated
(closed-form self-consistency plus an empirical Monte Carlo cross-check
against the real BKM07 simulator, both in the notebook); running the
actual benchmark grid across all three protocols is a natural next step
this layer enables but does not itself perform.

## 7. Model training and evaluation

Six classifiers per protocol (KNN, Logistic Regression, Random Forest,
SVM-RBF, XGBoost/HistGradientBoosting, and an unsupervised Isolation
Forest trained only on secure data), with Random Forest, SVM-RBF, and the
boosted-tree model tuned via 5-fold stratified `GridSearchCV` on ROC-AUC.
Evaluation reports ACC/AUC/F1 on a held-out 80/20 split — but this is
explicitly the *weakest* piece of evidence in the notebook (see Sections
8b-9 below), reported as a baseline to compare the more rigorous tests
against, not as the headline result.

### Model A/A+/B/C ablation

Rather than reporting one number for "the full feature set," a nested
ablation adds one information source at a time (QBER-only → + aggregate
statistics → + temporal structure → + decoy-state observables) and
reports **per-class FNR at a fixed 1% false-positive budget**, not just
AUC, with 95% CIs over repeated splits. This directly attributes
performance to a specific feature group instead of leaving "the model
uses everything" unexamined, and doubles as a physics correctness check:
PNS detection should jump specifically at the decoy-state step, and
intercept-resend detection should improve specifically at the temporal
step — if that pattern doesn't hold, something regressed upstream.

### Generalisation testing (E1-E4)

Four splits of increasing difficulty: the ordinary random split (weakest
evidence), unseen attack intensity, unseen distance range, and
leave-one-cell-out over a (distance × intensity) grid with a reported 95%
CI across cells. This replaced an earlier single train/test split
(`noise ≤ 4%` vs `≥ 8%`) that had become physically meaningless after the
distance range was corrected to 0-100km — a good example of a fix in one
place quietly invalidating an unrelated cell elsewhere in the notebook.

### N × window sensitivity

A separate grid over pulses-per-run (`N`) and number-of-windows (`W`)
quantifies exactly how much the temporal-feature block adds
(`ΔAUC = AUC(QBER+temporal) - AUC(QBER only)`) as a function of the
statistics actually available to compute those features — since a
per-window QBER estimate needs enough sifted bits before its own sampling
noise stops dominating the signal it's supposed to measure.

### Permutation importance, not impurity importance

Feature-importance plots use `sklearn.inspection.permutation_importance`
(drop in held-out ROC-AUC when one column is shuffled), not the
tree-based impurity importance scikit-learn provides by default —
impurity importance is biased toward continuous/high-cardinality features
(Strobl et al., 2007), which matters here because several features are
near-collinear by construction (`qber_total`/`qber_z`/`qber_x`/`h_qber`).
Permutation importance has its own bias under feature correlation
(Hooker, Mentch & Zhou, 2021), so the plots are documented as a ranking
among *groups* of correlated features, not a precise per-column
attribution.

## 8. Known limitations

- **Individual attacks only.** Coherent/collective attacks (Eve stores
  qubits and measures later using information revealed during sifting)
  are strictly stronger and not modelled.
- **No finite-key composable security.** BB84's `r_secure` is an
  asymptotic GLLP+decoy rate, used here as a channel-state *feature*, not
  a security certificate.
- **No detector-side attacks.** Detector blinding / efficiency-mismatch
  attacks act on the physical detector hardware, which this channel-level
  model cannot represent by construction.
- **Reduced dataset scale for runtime.** `samples_per_class=60` keeps the
  whole notebook runnable in a few minutes; the leave-one-cell-out
  confidence intervals are visibly wide at this scale.
- **E91's attack set excludes source/detector-hardware attacks.** An
  earlier version included PNS/detector-blinding analogues for E91, but
  their tell-tale features were sampled from label-conditioned
  distributions rather than genuinely simulated (trivially separable by
  construction) and were removed rather than kept as misleading results.

## 9. Running the notebook

Open `qkd_ml_v2.ipynb` and run top-to-bottom. Expect roughly 5-6 minutes
total, dominated by Section 4's dataset generation and Section 6's
hyperparameter tuning. `MPLBACKEND=Agg` is recommended for a headless/CI
run (all plots are also saved to `plots/`). Generated datasets are written
to `data/`.

## 10. Key references

- Ma, Qi, Zhao & Lo, "Practical decoy state for quantum key distribution,"
  *Phys. Rev. A* 72, 012326 (2005).
- Gobby, Yuan & Shields, "Quantum key distribution over 122 km of standard
  telecom fiber," *Appl. Phys. Lett.* 84, 3762 (2004).
- Lo, Ma & Chen, "Decoy state quantum key distribution," *PRL* 94, 230504
  (2005).
- Boyer, Kenigsberg & Mor, "Quantum key distribution with classical Bob,"
  *PRL* 99, 140501 (2007).
- Ekert, "Quantum cryptography based on Bell's theorem," *PRL* 67, 661
  (1991).
- Werner, "Quantum states with Einstein-Podolsky-Rosen correlations
  admitting a hidden-variable model," *Phys. Rev. A* 40, 4277 (1989).
- Kapoor & Narayanan, "Leakage and the reproducibility crisis in
  machine-learning-based science," *Patterns* 4, 100804 (2023).
- Strobl, Boulesteix, Zeileis & Hothorn, "Bias in random forest variable
  importance measures," *BMC Bioinformatics* 8:25 (2007).
- Hooker, Mentch & Zhou, "Unrestricted permutation forces extrapolation:
  variable importance requires at least one more model or a second
  opinion," *Stat. Comput.* 31:82 (2021).
- Brassard, Lütkenhaus, Mor & Sanders, "Limitations on practical quantum
  cryptography," *PRL* 85, 1330 (2000) — the PNS attack.
- Fuchs, Gisin, Griffiths, Niu & Peres, "Optimal eavesdropping in quantum
  cryptography," *Phys. Rev. A* 56, 1163 (1997) — entangling-probe attacks.
- Lydersen et al., "Hacking commercial quantum cryptography systems by
  tailored bright illumination," *Nature Photonics* 4, 686 (2010) —
  detector-blinding attacks (not modelled here; see Section 8).
