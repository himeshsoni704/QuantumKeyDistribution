import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE & OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
md(r"""# QKD Eavesdropping Detection with Machine Learning (v2)

**Title:** ML-Based Eavesdropping Detection in BB84 (Fully Quantum) vs BKM07 (Semi-Quantum) Key Distribution Protocols

---

## What this notebook does

Quantum Key Distribution (QKD) lets two parties — traditionally called **Alice** and **Bob** — share a secret cryptographic key in a way that any eavesdropper (**Eve**) inevitably disturbs the channel and gets detected. But *how* do you detect that disturbance reliably, especially when the channel already has background noise?

This notebook answers that question using **machine learning**:

1. We **simulate** photon-by-photon runs of two QKD protocols (BB84 and BKM07) under various attack scenarios.
2. We **extract statistical features** from each simulated run — things like error rates, their variance over time, and spectral properties.
3. We **train and compare six ML classifiers** to distinguish "secure channel" from "Eve is present."
4. We **tune** three of those models with 5-fold cross-validation and plot ROC curves + feature importances.

### The two protocols

| Protocol | Type | Key idea |
|---|---|---|
| **BB84** (Bennett & Brassard, 1984) | Fully quantum | Alice sends single photons in one of two bases; Eve's intercept forces a random re-preparation, causing detectable errors |
| **BKM07** (Boyer–Kenigsberg–Mor, 2007) | Semi-quantum | Bob is "classical" — he can only measure in the Z-basis or reflect. Eve must attack both the forward and return legs to learn anything |

### Improvements in v2 over v1

| Area | Change |
|---|---|
| Feature engineering | Added `jump_energy`, `spectral_entropy`, `autocorr_lag1` derived from the per-window QBER time-series |
| Classifiers | Replaced NumPy hand-rolled models with scikit-learn; added SVM-RBF and XGBoost |
| Hyperparameter tuning | 5-fold stratified CV for Boosted Trees, SVM-RBF **and** Random Forest (previously only boosted was tuned) |
| Adversarial test | Generalisation check: train on low-noise, test on high-noise examples |
| Output | Feature importance plot in addition to ROC curves |
''')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: SETUP
# ─────────────────────────────────────────────────────────────────────────────
md(r"""---
## Section 0 — Setup & Imports

We use:
- **NumPy / Matplotlib** for numerics and plotting
- **scikit-learn** for all ML models, pipelines, scaling, and evaluation
- **XGBoost** (optional) — the script gracefully falls back to scikit-learn's `HistGradientBoostingClassifier` if XGBoost is not installed

Run the cell below to install any missing packages, then import everything.
''')

code(r"""
# Uncomment if you need to install:
# !pip install scikit-learn xgboost --break-system-packages

import numpy as np
import matplotlib
matplotlib.use('Agg')          # use non-interactive backend (safe for notebooks too)
import matplotlib.pyplot as plt
import csv, os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

np.random.seed(42)
os.makedirs('data',  exist_ok=True)
os.makedirs('plots', exist_ok=True)

print(f"XGBoost available : {HAS_XGB}")
print("All imports OK.")
''')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: QUBIT PHYSICS
# ─────────────────────────────────────────────────────────────────────────────
md(r"""---
## Section 1 — Simulating Qubit Physics

### Qubit state representation

A qubit (quantum bit) is represented as a 2-element complex vector:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

We use two *bases*:

| Basis index | Name | $|0\rangle$-like state | $|1\rangle$-like state |
|---|---|---|---|
| 0 | **Z-basis** (rectilinear) | $|0\rangle = [1,\ 0]^T$ | $|1\rangle = [0,\ 1]^T$ |
| 1 | **X-basis** (diagonal) | $|+\rangle = \tfrac{1}{\sqrt{2}}[1,\ 1]^T$ | $|-\rangle = \tfrac{1}{\sqrt{2}}[1,\ -1]^T$ |

### Measurement (Born rule)

When Bob measures in basis $b$, the probability of getting outcome 0 is $|\langle b_0|\psi\rangle|^2$. The state then collapses to the measurement outcome.

### Noise model

`measure_qubit` already incorporates a **depolarizing channel**: with probability `noise_prob` the result is replaced by a uniformly random bit. This directly drives QBER — a higher noise level means more mismatches between Alice's sent bit and Bob's received bit, even without Eve.

> **Key insight:** Eve's intercept-resend attack *also* introduces extra errors (QBER ≈ 25% for 100% interception in BB84). The ML models learn to separate "noise-caused errors" from "Eve-caused errors" using the shape and temporal pattern of the error sequence, not just its average level.
''')

code(r"""
# ── Four fixed basis states ──────────────────────────────────────────────────
STATE_0    = np.array([1.0,  0.0], dtype=complex)
STATE_1    = np.array([0.0,  1.0], dtype=complex)
STATE_PLUS = np.array([1.0,  1.0], dtype=complex) / np.sqrt(2)
STATE_MINUS= np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)


def prepare_state(bit, basis):
    '''Map a classical bit (0 or 1) into a quantum state vector.

    basis=0 -> Z-basis (|0> / |1>)
    basis=1 -> X-basis (|+> / |->)
    '''
    if basis == 0:
        return STATE_0.copy() if bit == 0 else STATE_1.copy()
    else:
        return STATE_PLUS.copy() if bit == 0 else STATE_MINUS.copy()


def measure_qubit(state, basis, noise_prob):
    '''Measure a qubit in the given basis, subject to depolarizing noise.

    With probability `noise_prob` the channel corrupts the photon and we
    get a uniformly random bit (this is what drives QBER).
    Otherwise we use the Born-rule probability to decide the outcome.

    Returns
    -------
    (measured_bit, collapsed_state)
    '''
    # ── Noise: randomise the result ─────────────────────────────────────────
    if np.random.rand() < noise_prob:
        measured_bit = np.random.randint(0, 2)
        return measured_bit, prepare_state(measured_bit, basis)

    # ── No noise: use quantum probability ───────────────────────────────────
    if basis == 0:
        prob_0 = float(np.abs(np.dot(STATE_0.conj(), state)) ** 2)
    else:
        prob_0 = float(np.abs(np.dot(STATE_PLUS.conj(), state)) ** 2)

    prob_0 = float(np.clip(prob_0, 0.0, 1.0))
    measured_bit = 0 if np.random.rand() < prob_0 else 1
    return measured_bit, prepare_state(measured_bit, basis)


# ── Quick sanity check ───────────────────────────────────────────────────────
# Prepare |+⟩, measure in Z-basis many times -> should be ~50% each outcome
results = [measure_qubit(STATE_PLUS.copy(), basis=0, noise_prob=0.0)[0]
           for _ in range(2000)]
print(f"Measuring |+⟩ in Z-basis 2000 times:")
print(f"  P(0) ≈ {results.count(0)/2000:.3f}  (expected 0.500)")
print(f"  P(1) ≈ {results.count(1)/2000:.3f}  (expected 0.500)")
''')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ATTACK MODELS
# ─────────────────────────────────────────────────────────────────────────────
md(r"""---
## Section 2 — Protocol Simulations & Attack Models

### BB84 pulse simulation

Each call to `simulate_bb84_pulse` models one photon travelling from Alice to Bob:

```
Alice -> [photon in chosen basis] -> (Eve?) -> Bob measures
```

**Attack modes:**

| `eve_mode` | What Eve does | Effect on QBER |
|---|---|---|
| `'none'` | No attack | Only channel noise |
| `'intercept_resend'` | Eve measures in a random basis, re-sends | +25% QBER at 100% intensity (Eve guesses wrong basis 50% of the time, and half those wrong-basis measurements collapse to the wrong bit) |
| `'pns'` | Photon-Number-Splitting: Eve keeps an extra photon from multi-photon pulses | **Zero** extra QBER — this is why PNS is dangerous and why `multi_rate` is a feature |

### BKM07 pulse simulation

BKM07 is a *semi-quantum* protocol: Alice is fully quantum, Bob only has classical capabilities (measure in Z or reflect unchanged).

```
Alice -> [photon] -> Bob (SIFT or CTRL) -> [reflected photon] -> Alice
```

- **SIFT:** Bob measures in Z, re-prepares, sends back. Key bit established.
- **CTRL:** Bob reflects without measuring. Alice checks the return matches what she sent — disturbance here reveals Eve.

Eve must attack **both** legs (forward + return), so BKM07 offers two independent detection channels.
''')

code(r"""
def simulate_bb84_pulse(noise_prob, eve_mode, eve_intensity,
                        pns_active=False, mu=0.1):
    '''Simulate one BB84 photon transmission with optional eavesdropping.

    Parameters
    ----------
    noise_prob    : float  — depolarising noise probability per measurement
    eve_mode      : str    — 'none' | 'intercept_resend' | 'pns'
    eve_intensity : float  — probability Eve intercepts (intercept_resend mode)
    pns_active    : bool   — if True, model Poisson photon counts (weak coherent pulse)
    mu            : float  — mean photon number per pulse (WCP parameter)

    Returns dict with keys: bit_A, basis_A, basis_B, bit_B,
                            photon_n, pns_exploited, eve_info
    '''
    bit_A   = np.random.randint(0, 2)
    basis_A = np.random.randint(0, 2)
    state   = prepare_state(bit_A, basis_A)

    # Weak coherent pulses have a Poisson-distributed photon number.
    # Ideal single-photon sources would always send exactly 1.
    photon_n = np.random.poisson(mu) if pns_active else 1

    eve_info      = 0.0
    pns_exploited = False

    # ── Eve's action ─────────────────────────────────────────────────────────
    if eve_mode == 'pns' and pns_active and photon_n >= 2:
        # PNS: Eve silently keeps one photon. No collapse, no added error.
        pns_exploited = True
        eve_info = 1.0

    elif eve_mode == 'intercept_resend':
        if np.random.rand() < eve_intensity:
            # Eve guesses a basis and measures. If wrong basis -> 50% re-send error.
            basis_E = np.random.randint(0, 2)
            _, state = measure_qubit(state, basis_E, noise_prob=0.0)
            eve_info = 1.0 if basis_E == basis_A else 0.0

    # ── Bob measures ─────────────────────────────────────────────────────────
    basis_B = np.random.randint(0, 2)
    bit_B, _ = measure_qubit(state, basis_B, noise_prob)

    return {
        'bit_A': bit_A, 'basis_A': basis_A, 'basis_B': basis_B,
        'bit_B': bit_B, 'photon_n': photon_n,
        'pns_exploited': pns_exploited, 'eve_info': eve_info
    }



def simulate_bkm07_pulse(noise_prob, eve_mode, eve_fwd, eve_ret):
    '''Simulate one BKM07 round-trip pulse.

    Alice sends a photon -> Bob (SIFT or CTRL) -> reflected back -> Alice.
    Eve can attack on the forward leg (probability eve_fwd) and/or the
    return leg (probability eve_ret).

    Parameters
    ----------
    noise_prob : float — total channel noise (split evenly fwd/ret)
    eve_mode   : str   — 'none' | 'symmetric' | 'asymmetric'
    eve_fwd    : float — Eve's interception probability on the forward leg
    eve_ret    : float — Eve's interception probability on the return leg
    '''
    fwd_noise = noise_prob / 2.0   # noise budget split equally
    ret_noise = noise_prob / 2.0

    bit_A   = np.random.randint(0, 2)
    basis_A = np.random.randint(0, 2)
    state   = prepare_state(bit_A, basis_A)

    # ── Eve on the forward leg ────────────────────────────────────────────────
    if eve_mode != 'none' and np.random.rand() < eve_fwd:
        basis_Ef = np.random.randint(0, 2)
        _, state = measure_qubit(state, basis_Ef, noise_prob=0.0)

    # ── Bob: SIFT (measure + re-prepare) or CTRL (reflect unchanged) ──────────
    bob_mode = np.random.choice(['SIFT', 'CTRL'])
    bit_B = None
    if bob_mode == 'SIFT':
        bit_B, state = measure_qubit(state, 0, fwd_noise)
        state = prepare_state(bit_B, 0)   # classical re-preparation

    # ── Eve on the return leg ─────────────────────────────────────────────────
    if eve_mode != 'none' and np.random.rand() < eve_ret:
        basis_Er = np.random.randint(0, 2)
        _, state = measure_qubit(state, basis_Er, noise_prob=0.0)

    # ── Alice measures the returning photon ───────────────────────────────────
    if bob_mode == 'SIFT':
        bit_A_final, _ = measure_qubit(state, 0, ret_noise)
    else:
        bit_A_final, _ = measure_qubit(state, basis_A, fwd_noise + ret_noise)

    return {
        'bit_A': bit_A, 'basis_A': basis_A, 'bob_mode': bob_mode,
        'bit_B': bit_B, 'bit_A_final': bit_A_final
    }
''')

# ── Demo: QBER rises with Eve's interception rate ────────────────────────────
eve_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
print("BB84 QBER vs Eve intercept intensity (noise_prob=0.02, N=5000):")
print(f"  {'Eve intensity':>14}  {'QBER':>6}")
for intensity in eve_levels:
    pulses = [simulate_bb84_pulse(0.02, 'intercept_resend', intensity)
              for _ in range(5000)]
    sifted = [(p['bit_A'], p['bit_B']) for p in pulses
              if p['basis_A'] == p['basis_B']]
    qber = sum(a != b for a, b in sifted) / len(sifted)
    print(f"  {intensity:>14.1f}  {qber:>6.3f}")
''')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
md(r"""---
## Section 3 — Feature Engineering

A single run of N=2000 pulses produces too many individual events to feed directly into a classifier. Instead we compress each run into a **feature vector** — a small set of summary statistics that capture what matters.

### Why temporal features?

Average QBER alone can be ambiguous: channel noise and a mild attack can produce similar averages. But their **time profiles** differ:

- **Uniform noise** -> QBER is roughly constant across time windows (low variance, low jump energy).
- **Intercept-resend attack** -> Eve's interception is random-in-time, creating bursts and correlations in the error sequence.
- **PNS attack** -> QBER barely changes, but `multi_rate` (fraction of multi-photon pulses) is a tell.

### Feature set

#### BB84 (10 features)

| Feature | Formula / idea |
|---|---|
| `qber_z` | Error rate in Z-basis sifted rounds |
| `qber_x` | Error rate in X-basis sifted rounds |
| `qber_total` | Overall sifted error rate |
| `qber_variance` | Variance of per-window QBER — high = unstable/bursty |
| `jump_energy` | $\sum_i (\text{QBER}_i - \text{QBER}_{i-1})^2$ — penalises sudden jumps |
| `spectral_entropy` | Normalised Shannon entropy of QBER power spectrum (0=structured, 1=noise-like) |
| `autocorr_lag1` | Lag-1 autocorrelation — positive if errors cluster in time |
| `holevo_ie` | $h(\text{QBER}_\text{total})$ — rough upper bound on Eve's information |
| `sifted_rate` | Fraction of pulses where Alice & Bob chose the same basis |
| `multi_rate` | Fraction of pulses with ≥2 photons (PNS detector) |

#### BKM07 (12 features)

BKM07 has separate forward and return legs, so we get more error channels:

| Feature | What it measures |
|---|---|
| `qber_zs` | Forward-leg error rate (Z-basis SIFT rounds) |
| `qber_zsr` | Return-leg error rate (Z-basis SIFT rounds) |
| `qber_xsr` | Return-leg error rate (X-basis SIFT rounds) |
| `qber_zc` | Z-basis CTRL-round error rate |
| `qber_xc` | X-basis CTRL-round error rate |
| `asymmetry` | $|\text{qber\_zsr} - \text{qber\_zs}|$ — signature of return-heavy attack |
| `xctrl_variance` | Variance of per-window X-CTRL QBER |
| `jump_energy` | Temporal jump energy of X-CTRL QBER trace |
| `spectral_entropy` | Spectral entropy of X-CTRL QBER trace |
| `autocorr_lag1` | Lag-1 autocorrelation of X-CTRL QBER trace |
| `holevo_ie` | $h(\text{avg CTRL QBER})$ |
| `sifted_rate` | Fraction of pulses that contributed to the key |
''')

code(r"""
# ─── Helper: binary entropy (Shannon, base-2) ────────────────────────────────
def _binary_entropy(p):
    '''H(p) = -p log2(p) - (1-p) log2(1-p). Used for the Holevo bound.'''
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def holevo_bound(qber):
    return _binary_entropy(qber)


# ─── Three temporal-shape features ───────────────────────────────────────────

def _jump_energy(x):
    '''Sum of squared consecutive differences.
    High  -> the QBER trace jumps around a lot (bursty, attack-like).
    Low   -> smooth, slowly-varying trace (channel noise).
    '''
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.diff(x) ** 2))


def _spectral_entropy(x):
    '''Normalised Shannon entropy of the power spectrum.

    Intuition:
      A white-noise-like QBER trace has energy spread evenly across all
      frequencies  ->  high entropy (close to 1).
      A periodic or structured disturbance concentrates energy in a few
      frequencies  ->  low entropy (close to 0).

    Steps:
      1. Zero-mean the sequence.
      2. Compute FFT power spectrum.
      3. Normalise to a probability distribution.
      4. Compute Shannon entropy, normalised by log2(N_freq).
    '''
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    x = x - x.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    psd = spectrum / total
    psd = psd[psd > 0]
    if len(psd) < 2:
        return 0.0
    ent = -np.sum(psd * np.log2(psd))
    return float(ent / np.log2(len(psd)))


def _autocorr_lag1(x):
    '''Lag-1 autocorrelation.

    Near 0   -> uncorrelated (pure channel noise).
    Positive -> high-error windows tend to cluster (sustained attack or
               slowly-varying noise source).
    Computed as the normalised inner product of x[:-1] and x[1:].
    '''
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


# ── Quick visual: compare temporal features for secure vs attacked ────────────
def _get_window_qbers(pulses, window=200):
    wqs = []
    for start in range(0, len(pulses), window):
        chunk = pulses[start:start + window]
        tot, err = 0, 0
        for p in chunk:
            if p['basis_A'] == p['basis_B']:
                tot += 1
                err += (p['bit_A'] != p['bit_B'])
        wqs.append(err / tot if tot > 0 else 0.0)
    return wqs

print("Computing window-level QBER traces for 2000 pulses each…")
secure_pulses = [simulate_bb84_pulse(0.03, 'none', 0.0) for _ in range(2000)]
attack_pulses = [simulate_bb84_pulse(0.03, 'intercept_resend', 0.15) for _ in range(2000)]

wq_sec = _get_window_qbers(secure_pulses)
wq_att = _get_window_qbers(attack_pulses)

print(f"\nSecure  — jump_energy={_jump_energy(wq_sec):.5f}  "
      f"spectral_entropy={_spectral_entropy(wq_sec):.3f}  "
      f"autocorr_lag1={_autocorr_lag1(wq_sec):.3f}")
print(f"Attacked — jump_energy={_jump_energy(wq_att):.5f}  "
      f"spectral_entropy={_spectral_entropy(wq_att):.3f}  "
      f"autocorr_lag1={_autocorr_lag1(wq_att):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, wq, title, color in [
        (axes[0], wq_sec, 'Secure (no Eve)', '#2563EB'),
        (axes[1], wq_att, 'Under intercept-resend attack', '#DC2626')]:
    ax.plot(wq, color=color, linewidth=1.5)
    ax.axhline(np.mean(wq), color='black', linestyle='--', linewidth=1,
               label=f'mean={np.mean(wq):.3f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Window index'); ax.set_ylabel('QBER')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Per-window QBER trace: secure vs attacked', fontsize=12)
plt.tight_layout()
plt.savefig('plots/qber_traces.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to plots/qber_traces.png")
''')

code(r'''
# ─── Full feature-extraction functions ───────────────────────────────────────

def collect_bb84_features(N, noise_prob, eve_mode, eve_intensity,
                           window=200, pns_active=False, mu=0.1):
    '''Run N BB84 pulses and compress them into a 10-element feature vector.

    The raw pulses are first split into time-windows of `window` pulses each.
    The per-window QBER sequence is then fed into the temporal-shape features.
    '''
    pulses = [simulate_bb84_pulse(noise_prob, eve_mode, eve_intensity,
                                   pns_active, mu)
              for _ in range(N)]

    sz, se_z = 0, 0    # Z-basis sifted count / errors
    sx, se_x = 0, 0    # X-basis sifted count / errors
    multi_photon_count = 0

    for p in pulses:
        if p['photon_n'] >= 2:
            multi_photon_count += 1
        if p['basis_A'] == p['basis_B']:       # sifted round
            if p['basis_A'] == 0:
                sz  += 1
                se_z += int(p['bit_A'] != p['bit_B'])
            else:
                sx  += 1
                se_x += int(p['bit_A'] != p['bit_B'])

    qber_z     = se_z / sz if sz > 0 else 0.0
    qber_x     = se_x / sx if sx > 0 else 0.0
    qber_total = (se_z + se_x) / (sz + sx) if (sz + sx) > 0 else 0.0
    sifted_rate = (sz + sx) / N
    multi_rate  = multi_photon_count / N

    window_qbers = []
    for start in range(0, N, window):
        w = pulses[start:start + window]
        wsz, wse = 0, 0
        for p in w:
            if p['basis_A'] == p['basis_B']:
                wsz += 1
                wse += int(p['bit_A'] != p['bit_B'])
        window_qbers.append(wse / wsz if wsz > 0 else 0.0)

    return {
        'qber_z':           qber_z,
        'qber_x':           qber_x,
        'qber_total':       qber_total,
        'qber_variance':    float(np.var(window_qbers)),
        'jump_energy':      _jump_energy(window_qbers),
        'spectral_entropy': _spectral_entropy(window_qbers),
        'autocorr_lag1':    _autocorr_lag1(window_qbers),
        'holevo_ie':        holevo_bound(qber_total),
        'sifted_rate':      sifted_rate,
        'multi_rate':       multi_rate,
    }


def collect_bkm07_features(N, noise_prob, eve_mode, eve_fwd, eve_ret,
                            window=200):
    '''Run N BKM07 pulses and compress into a 12-element feature vector.

    BKM07 has richer structure: we track errors on forward SIFT, return SIFT,
    Z-CTRL, and X-CTRL channels separately. The `asymmetry` feature captures
    attacks that are heavier on one leg than the other.
    '''
    pulses = [simulate_bkm07_pulse(noise_prob, eve_mode, eve_fwd, eve_ret)
              for _ in range(N)]

    z_sft_t, z_sft_e, z_sft_ret_e = 0, 0, 0
    x_sft_t, x_sft_ret_e           = 0, 0
    z_ctrl_t, z_ctrl_e             = 0, 0
    x_ctrl_t, x_ctrl_e             = 0, 0

    for p in pulses:
        if p['bob_mode'] == 'SIFT':
            if p['basis_A'] == 0:
                z_sft_t     += 1
                z_sft_e     += int(p['bit_A'] != p['bit_B'])
                z_sft_ret_e += int(p['bit_B'] != p['bit_A_final'])
            else:
                x_sft_t     += 1
                x_sft_ret_e += int(p['bit_B'] != p['bit_A_final'])
        else:   # CTRL
            if p['basis_A'] == 0:
                z_ctrl_t += 1
                z_ctrl_e += int(p['bit_A'] != p['bit_A_final'])
            else:
                x_ctrl_t += 1
                x_ctrl_e += int(p['bit_A'] != p['bit_A_final'])

    qber_zs  = z_sft_e     / z_sft_t  if z_sft_t  > 0 else 0.0
    qber_zsr = z_sft_ret_e / z_sft_t  if z_sft_t  > 0 else 0.0
    qber_xsr = x_sft_ret_e / x_sft_t  if x_sft_t  > 0 else 0.0
    qber_zc  = z_ctrl_e    / z_ctrl_t if z_ctrl_t > 0 else 0.0
    qber_xc  = x_ctrl_e    / x_ctrl_t if x_ctrl_t > 0 else 0.0

    asymmetry = abs(qber_zsr - qber_zs)

    window_xctrl = []
    for start in range(0, N, window):
        w = pulses[start:start + window]
        wt, we = 0, 0
        for p in w:
            if p['bob_mode'] == 'CTRL' and p['basis_A'] == 1:
                wt += 1
                we += int(p['bit_A'] != p['bit_A_final'])
        window_xctrl.append(we / wt if wt > 0 else 0.0)

    qber_ctrl_avg = (qber_zc + qber_xc) / 2.0

    return {
        'qber_zs':          qber_zs,
        'qber_zsr':         qber_zsr,
        'qber_xsr':         qber_xsr,
        'qber_zc':          qber_zc,
        'qber_xc':          qber_xc,
        'asymmetry':        asymmetry,
        'xctrl_variance':   float(np.var(window_xctrl)),
        'jump_energy':      _jump_energy(window_xctrl),
        'spectral_entropy': _spectral_entropy(window_xctrl),
        'autocorr_lag1':    _autocorr_lag1(window_xctrl),
        'holevo_ie':        holevo_bound(qber_ctrl_avg),
        'sifted_rate':      z_sft_t / N,
    }


print("Feature-extraction functions defined.")
''')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
md(r"""---
## Section 4 — Generating the Labelled Dataset

We generate `samples_per_class` independent simulated runs for each of the six scenario classes:

| Protocol | Label | Scenario |
|---|---|---|
| BB84 | 0 | Secure (no Eve) |
| BB84 | 1 | Intercept-resend attack |
| BB84 | 2 | Photon-number-splitting (PNS) attack |
| BKM07 | 0 | Secure (no Eve) |
| BKM07 | 1 | Symmetric attack (equal forward/return intensity) |
| BKM07 | 3 | Asymmetric attack (strong return, weak forward) |

For each run, a random channel noise level is drawn from $\text{Uniform}(0, 0.10)$ and (for attacked runs) a random Eve intensity. This variation is deliberate: it forces the models to learn **attack signatures** rather than just "high QBER = bad."

The datasets are also written to `data/bb84_dataset.csv` and `data/bkm07_dataset.csv`.

> **Note on training time:** With `samples_per_class=400` and `N=2000`, this generates 1200 BB84 samples and 1200 BKM07 samples. Each sample runs 2000 quantum pulse simulations. Expect 2–5 minutes depending on your hardware.
''')

code(r"""
def generate_datasets(samples_per_class=400, N=2000):
    '''Simulate QKD runs for all attack classes and return feature arrays.

    Returns
    -------
    bb84_arr : np.ndarray, shape (3*samples_per_class, 13)
        Columns: 10 features + noise_level + eve_intensity + label
    bkm_arr  : np.ndarray, shape (3*samples_per_class, 16)
        Columns: 12 features + noise_level + eve_fwd + eve_ret + label
    bb84_header, bkm_header : list[str]
    '''
    bb84_rows, bkm_rows = [], []

    # ── BB84: secure ──────────────────────────────────────────────────────────
    print("  [BB84] Generating secure samples (label 0) …")
    for _ in range(samples_per_class):
        e = np.random.uniform(0.0, 0.10)
        f = collect_bb84_features(N, e, 'none', 0.0)
        bb84_rows.append([f['qber_z'], f['qber_x'], f['qber_total'],
                          f['qber_variance'], f['jump_energy'],
                          f['spectral_entropy'], f['autocorr_lag1'],
                          f['holevo_ie'], f['sifted_rate'], f['multi_rate'],
                          e, 0.0, 0])

    # ── BB84: intercept-resend ────────────────────────────────────────────────
    print("  [BB84] Generating intercept-resend samples (label 1) …")
    for _ in range(samples_per_class):
        e  = np.random.uniform(0.0, 0.10)
        di = np.random.uniform(0.03, 0.20)   # Eve intercepts 3–20% of pulses
        f  = collect_bb84_features(N, e, 'intercept_resend', di)
        bb84_rows.append([f['qber_z'], f['qber_x'], f['qber_total'],
                          f['qber_variance'], f['jump_energy'],
                          f['spectral_entropy'], f['autocorr_lag1'],
                          f['holevo_ie'], f['sifted_rate'], f['multi_rate'],
                          e, di, 1])

    # ── BB84: PNS attack ──────────────────────────────────────────────────────
    print("  [BB84] Generating PNS attack samples (label 2) …")
    for _ in range(samples_per_class):
        e  = np.random.uniform(0.0, 0.10)
        mu = np.random.uniform(0.08, 0.25)   # mean photon number
        f  = collect_bb84_features(N, e, 'pns', 0.0, pns_active=True, mu=mu)
        bb84_rows.append([f['qber_z'], f['qber_x'], f['qber_total'],
                          f['qber_variance'], f['jump_energy'],
                          f['spectral_entropy'], f['autocorr_lag1'],
                          f['holevo_ie'], f['sifted_rate'], f['multi_rate'],
                          e, mu, 2])

    # ── BKM07: secure ─────────────────────────────────────────────────────────
    print("  [BKM07] Generating secure samples (label 0) …")
    for _ in range(samples_per_class):
        e = np.random.uniform(0.0, 0.10)
        f = collect_bkm07_features(N, e, 'none', 0.0, 0.0)
        bkm_rows.append([f['qber_zs'], f['qber_zsr'], f['qber_xsr'],
                         f['qber_zc'], f['qber_xc'],
                         f['asymmetry'], f['xctrl_variance'],
                         f['jump_energy'], f['spectral_entropy'], f['autocorr_lag1'],
                         f['holevo_ie'], f['sifted_rate'],
                         e, 0.0, 0.0, 0])

    # ── BKM07: symmetric attack ───────────────────────────────────────────────
    print("  [BKM07] Generating symmetric attack samples (label 1) …")
    for _ in range(samples_per_class):
        e  = np.random.uniform(0.0, 0.10)
        di = np.random.uniform(0.03, 0.20)
        f  = collect_bkm07_features(N, e, 'symmetric', di, di)
        bkm_rows.append([f['qber_zs'], f['qber_zsr'], f['qber_xsr'],
                         f['qber_zc'], f['qber_xc'],
                         f['asymmetry'], f['xctrl_variance'],
                         f['jump_energy'], f['spectral_entropy'], f['autocorr_lag1'],
                         f['holevo_ie'], f['sifted_rate'],
                         e, di, di, 1])

    # ── BKM07: asymmetric attack ──────────────────────────────────────────────
    print("  [BKM07] Generating asymmetric attack samples (label 3) …")
    for _ in range(samples_per_class):
        e      = np.random.uniform(0.0, 0.10)
        di_fwd = np.random.uniform(0.01, 0.08)    # light forward interception
        di_ret = np.random.uniform(0.10, 0.30)    # heavy return interception
        f      = collect_bkm07_features(N, e, 'asymmetric', di_fwd, di_ret)
        bkm_rows.append([f['qber_zs'], f['qber_zsr'], f['qber_xsr'],
                         f['qber_zc'], f['qber_xc'],
                         f['asymmetry'], f['xctrl_variance'],
                         f['jump_energy'], f['spectral_entropy'], f['autocorr_lag1'],
                         f['holevo_ie'], f['sifted_rate'],
                         e, di_fwd, di_ret, 3])

    # ── Write CSVs ────────────────────────────────────────────────────────────
    bb84_header = ['qber_z','qber_x','qber_total','qber_variance',
                   'jump_energy','spectral_entropy','autocorr_lag1',
                   'holevo_ie','sifted_rate','multi_rate',
                   'noise_level','eve_intensity','label']
    bkm_header  = ['qber_zs','qber_zsr','qber_xsr','qber_zc','qber_xc',
                   'asymmetry','xctrl_variance',
                   'jump_energy','spectral_entropy','autocorr_lag1',
                   'holevo_ie','sifted_rate',
                   'noise_level','eve_fwd','eve_ret','label']

    with open('data/bb84_dataset.csv', 'w', newline='') as fh:
        csv.writer(fh).writerows([bb84_header] + bb84_rows)
    with open('data/bkm07_dataset.csv', 'w', newline='') as fh:
        csv.writer(fh).writerows([bkm_header] + bkm_rows)

    return (np.array(bb84_rows), np.array(bkm_rows),
            bb84_header, bkm_header)


print("generate_datasets() defined — running now (this takes a few minutes) …")
bb84_arr, bkm_arr, bb84_hdr, bkm_hdr = generate_datasets(
    samples_per_class=400, N=2000)

print(f"\nBB84 dataset shape : {bb84_arr.shape}   "
      f"(label counts: {dict(zip(*np.unique(bb84_arr[:,-1].astype(int), return_counts=True)))})")
print(f"BKM07 dataset shape: {bkm_arr.shape}   "
      f"(label counts: {dict(zip(*np.unique(bkm_arr[:,-1].astype(int), return_counts=True)))})")
print("CSVs saved to data/")
''')

# ... (rest of file omitted in this message for brevity; created full content in the file)
