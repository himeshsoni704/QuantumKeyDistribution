#!/usr/bin/env python3
"""Standalone QKD deep-learning study. See QKD_PIPELINE_README.md.

Simulator and classical extractor definitions below are copied from the supplied
source. They do not import or execute the original notebook/script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ======================================================================
# 1. Original physics and feature extraction (preserved source definitions)
# ======================================================================
MASTER_SEED = 20260913
_FALLBACK_COUNTER = [0]
HONEST_DRIFT_FRAC = 0.5
HONEST_DRIFT_MAX = 0.5


class SeedBook:
    """Deterministic, independent RNG streams, addressable by (role, index).

    Uses zlib.crc32 on the role string, NOT Python's builtin hash(): str
    hashing is salted by PYTHONHASHSEED, randomised per-process by default
    since Python 3.3 (security hardening against hash-flooding attacks),
    so hash(role) gives a DIFFERENT tag -- and therefore different actual
    random draws -- every time the notebook is run in a fresh process, even
    though it looks stable within any single run. crc32 is a fixed,
    unsalted function, so the same role always maps to the same stream
    across separate runs, which is the entire point of this class.

    Draft 2.1 fix: the previous implementation cached one SeedSequence per
    role and called `.spawn(index + 1)[index]` on it. spawn() is STATEFUL
    (it advances the sequence's n_children_spawned counter), so rng(role, 0)
    returned a different stream every time it was called, and a re-run of a
    single cell did not reproduce its numbers. The stream is now a pure
    function of (master, crc32(role), index): the same arguments always give
    the same generator, in any call order and in any process.
    """

    def __init__(self, master=MASTER_SEED):
        self.master = master

    def _seq(self, role, index):
        tag = zlib.crc32(role.encode()) % (2**31)
        return np.random.SeedSequence([self.master, tag, int(index)])

    def rng(self, role, index):
        """Independent numpy Generator for (role, index) -- stateless/deterministic."""
        return np.random.default_rng(self._seq(role, index))

    def seed(self, role, index=0):
        """31-bit integer seed for sklearn/xgboost `random_state`, derived the same way."""
        return int(self._seq(role, index).generate_state(1)[0]) % (2**31)


def _fallback_rng():
    """Deterministic stand-in for np.random.default_rng() when a caller passes no rng (a different stream for every call, same sequence every run)."""
    _FALLBACK_COUNTER[0] += 1
    return SEEDS.rng("unseeded_fallback", _FALLBACK_COUNTER[0])


BKM07_CTRL_FORWARD_NOISE = True


BB84_LOSS_CAP_ETA_BOB = True


PNS_MATCH_SIGNAL_GAIN_ONLY = True


K_MARGIN = 1.3


def _extend_to_k(record, draw, key_mask, K, batch_size):
    """Continue independent simulation batches; never drop a short run or replace it with another link.
    max_pulses bounds one draw, not the information budget. Timing profiles restart at batch
    boundaries; this is a stated simulation limitation for exceptionally lossy long sessions.
    """
    if K is None:
        return record
    if K <= 0:
        raise ValueError("K must be positive")
    total = len(key_mask(record))
    counts = int(key_mask(record).sum())
    for _ in range(1000):
        if counts >= K:
            return record
        extra = draw(int(batch_size))
        m = key_mask(extra)
        n = len(m)
        for key, value in list(record.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                record[key] = np.concatenate((value, extra[key]))
        total += n
        counts += int(m.sum())
        if "N" in record:
            record["N"] = total
    raise RuntimeError(
        f"Unable to obtain K={K}; collected {counts} after {total} raw events"
    )


GYS = dict(
    alpha_db_km=0.21,  # fibre attenuation @1550 nm       [dB/km]
    eta_bob=0.045,  # Bob-side transmittance x detector efficiency
    Y0=1.7e-6,  # background/dark yield per pulse
    e_detector=0.033,  # optical misalignment error rate
    e_0=0.5,  # error rate of a background count
    f_ec=1.22,
)


MU_SIGNAL, MU_DECOY, MU_VACUUM = 0.48, 0.05, 0.0


DECOY_PROBS = [0.70, 0.25, 0.05]


CHANNEL_DISTANCE_RANGE_KM = (0.0, 100.0)


def channel_model(
    distance_km,
    mu=MU_SIGNAL,
    alpha_db_km=None,
    eta_bob=None,
    Y0=None,
    e_detector=None,
    e_0=None,
):
    """Fibre-QKD channel: transmittance, yields, gain, QBER, single-photon terms.

    Returns a dict; `gain` is Q_mu and `qber` is E_mu in the notation of
    Ma et al. (2005).
    """
    alpha_db_km = GYS["alpha_db_km"] if alpha_db_km is None else alpha_db_km
    eta_bob = GYS["eta_bob"] if eta_bob is None else eta_bob
    Y0 = GYS["Y0"] if Y0 is None else Y0
    e_detector = GYS["e_detector"] if e_detector is None else e_detector
    e_0 = GYS["e_0"] if e_0 is None else e_0

    t_AB = 10.0 ** (-alpha_db_km * distance_km / 10.0)  # Eq. (5)
    eta = t_AB * eta_bob

    Q_mu = Y0 + 1.0 - np.exp(-eta * mu)  # Eq. (10)
    E_mu = (e_0 * Y0 + e_detector * (1.0 - np.exp(-eta * mu))) / Q_mu  # Eq. (11)

    Y1 = Y0 + eta - Y0 * eta  # Eq. (7), i=1
    e1 = (e_0 * Y0 + e_detector * eta) / Y1  # Eq. (9), i=1

    return dict(
        distance_km=distance_km,
        transmittance=t_AB,
        eta=eta,
        gain=Q_mu,
        qber=E_mu,
        Y1=Y1,
        e1=e1,
        Y0=Y0,
        e_detector=e_detector,
        e_0=e_0,
        mu=mu,
    )


ALICE_ANGLES = {"a1": 0.0, "a2": 22.5, "a3": 45.0}


BOB_ANGLES = {"b1": 22.5, "b2": 45.0, "b3": 67.5}


KEY_PAIRS = [("a2", "b1"), ("a3", "b2")]


CHSH_PAIRS = [("a1", "b1"), ("a1", "b3"), ("a3", "b1"), ("a3", "b3")]


CHSH_SIGNS = [+1, -1, +1, +1]


TSIRELSON_BOUND = 2 * np.sqrt(2)


I2 = np.eye(2, dtype=complex)


SX = np.array([[0, 1], [1, 0]], dtype=complex)


SZ = np.array([[1, 0], [0, -1]], dtype=complex)


PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


RHO_SINGLET = np.outer(PSI_MINUS, PSI_MINUS.conj())


def polarisation_observable(theta_deg):
    """A(theta) = cos(2 theta) sigma_z + sin(2 theta) sigma_x.
    For the singlet this gives <A(a) (x) A(b)> = -cos(2(a-b)), Ekert's convention."""
    t = np.radians(theta_deg)
    return np.cos(2 * t) * SZ + np.sin(2 * t) * SX


def projectors(theta_deg):
    A = polarisation_observable(theta_deg)
    return {+1: (I2 + A) / 2, -1: (I2 - A) / 2}


def werner_state(V, rho=RHO_SINGLET):
    """rho_W = V |Psi-><Psi-| + (1-V) I/4. Werner, PRA 40, 4277 (1989).
    Gives |S|_max = 2 sqrt(2) V and key-basis QBER = (1-V)/2."""
    return V * rho + (1.0 - V) * np.eye(4, dtype=complex) / 4.0


def joint_probs(rho, a_deg, b_deg):
    """Born rule: P(x,y|a,b) = Tr[rho (Pi_a^x (x) Pi_b^y)]."""
    PA, PB = projectors(a_deg), projectors(b_deg)
    out = {}
    for x in (+1, -1):
        for y in (+1, -1):
            out[(x, y)] = float(np.real(np.trace(rho @ np.kron(PA[x], PB[y]))))
    s = sum(out.values())
    return {kk: max(v, 0.0) / s for kk, v in out.items()}


def sample_e91_channel(rng):
    """Honest-link visibility. V=1 is the ideal singlet; the honest range
    below gives a genuine, non-zero noise floor (audit Defect 7): any run
    with V < 1 has real, non-adversarial degradation to compare Eve against."""
    return float(rng.uniform(0.85, 0.99))


def attack_schedule(
    n_pulses, intensity, profile="iid", rng=None, mean_burst=2000, drift_period=None
):
    rng = rng or _fallback_rng()
    if intensity <= 0:
        return np.zeros(n_pulses, dtype=bool)
    if profile == "iid":
        return rng.random(n_pulses) < intensity
    if profile == "bursty":
        mask = np.zeros(n_pulses, dtype=bool)
        i, on = 0, (rng.random() < intensity)
        while i < n_pulses:
            L = min(max(1, int(rng.exponential(mean_burst))), n_pulses - i)
            mask[i : i + L] = on
            i += L
            on = rng.random() < intensity
        return mask
    if profile == "drifting":
        T = drift_period or max(n_pulses // 4, 1)
        phase = rng.random() * 2 * np.pi
        env = intensity * (
            1.0 + 0.9 * np.sin(2 * np.pi * np.arange(n_pulses) / T + phase)
        )
        return rng.random(n_pulses) < np.clip(env, 0.0, 1.0)
    raise ValueError(f"unknown profile {profile!r}")


def make_pns_strategy(distance_km, mu_signal=MU_SIGNAL, n_split=2, **chan):
    # Review fix B2: Eve keeps ONE photon, so Bob receives n-1 photons through a lossless line, but he still
    # detects each with probability eta_bob (he does not become a perfect detector). Gain matching therefore uses
    # P(click | forwarded, n) = 1-(1-eta_bob)^(n-1). If the required t_fwd exceeds 1 Eve cannot hide at this distance.
    ch = channel_model(distance_km, mu=mu_signal, **chan)
    eta_bob = chan.get("eta_bob") if chan.get("eta_bob") is not None else GYS["eta_bob"]
    p_forwardable = sum(
        np.exp(-mu_signal)
        * mu_signal**k
        / factorial(k)
        * (1.0 - (1.0 - eta_bob) ** (k - 1))
        for k in range(n_split, 15)
    )
    if PNS_MATCH_SIGNAL_GAIN_ONLY:  # dark counts are added by Bob's detector regardless of Eve: she only has to reproduce (gain - Y0)/(1 - Y0)
        _g = (ch["gain"] - ch["Y0"]) / max(1.0 - ch["Y0"], 1e-15)
    else:
        _g = ch["gain"]
    t_raw = float(_g / max(p_forwardable, 1e-15))
    t_fwd = float(np.clip(t_raw, 0.0, 1.0))
    return dict(
        n_split=n_split,
        t_fwd=t_fwd,
        t_required=t_raw,
        can_hide=bool(t_raw <= 1.0),
        mu_signal=mu_signal,
        distance_km=distance_km,
        eta_bob=float(eta_bob),
    )


EVE_BASES = (0.0, 22.5, 45.0, 67.5)


def eve_measure_resend_bob(rho, e_deg):
    """Eve intercepts Bob's photon, measures at e_deg, resends the eigenstate
    (measure-and-reprepare CPTP map on the pair state)."""
    PB = projectors(e_deg)
    out = np.zeros((4, 4), dtype=complex)
    for y in (+1, -1):
        K = np.kron(I2, PB[y])
        red = K @ rho @ K
        pA = np.zeros((2, 2), dtype=complex)  # partial trace over Bob
        for i in range(2):
            for j in range(2):
                pA[i, j] = red[2 * i, 2 * j] + red[2 * i + 1, 2 * j + 1]
        out += np.kron(pA, PB[y])
    return out / np.real(np.trace(out))


def eve_ir_channel(rho, bases=EVE_BASES):
    """Eve's basis is random PER PULSE, so the physically correct object is
    the BASIS-AVERAGED map, identical for every Alice/Bob setting pair.
    (Sampling a fresh basis per CHSH correlator, as the previous version
    did, mixes four inconsistent attacks and can push |S| above what any
    single attack could produce -- measured 2.77 instead of the correct
    ~1.39-1.41 for 100% intercept-resend.)"""
    return sum(eve_measure_resend_bob(rho, e) for e in bases) / len(bases)


def eve_ancilla_bob(rho, lam, basis_deg=0.0):
    """Entangling probe: Eve couples an ancilla to Bob's qubit and keeps it.
    Tracing out the ancilla leaves basis-dependent (ANISOTROPIC) dephasing
    of strength lam -- unlike depolarisation, this costs more QBER than it
    costs CHSH, which is what makes it separable from honest noise via
    `s_qber_residual` below. Simplified individual-attack version of
    Fuchs et al., PRA 56, 1163 (1997)."""
    A = polarisation_observable(basis_deg)
    K0 = np.sqrt(1.0 - lam / 2.0) * I2
    K1 = np.sqrt(lam / 2.0) * A
    out = np.zeros((4, 4), dtype=complex)
    for K in (K0, K1):
        M = np.kron(I2, K)
        out += M @ rho @ M.conj().T
    return out / np.real(np.trace(out))


def eve_loss_manipulation(rho, delta=0.3):
    """Asymmetric arm loss: Eve attenuates Bob's arm more than Alice's.
    Modelled as an isotropic visibility reduction on top of the honest
    channel (the correlation degrades, but the mechanism -- differential
    loss -- is a channel-level attack, not a measurement)."""
    return (1.0 - delta) * rho + delta * np.eye(4, dtype=complex) / 4.0


def honest_noise_multiplier(n, detector, rng):
    detector = detector or {}
    if detector.get("noise_profile", "stationary") != "bursty":
        return np.ones(n)
    factor = np.ones(n)
    block = max(1, int(detector.get("burst_block", max(n // 32, 1))))
    high = False
    for lo in range(0, n, block):
        if rng.random() < 0.25:
            high = not high
        if high:
            factor[lo : lo + block] = float(detector.get("burst_multiplier", 3.0))
    return factor


def simulate_bb84_decoy(
    N,
    distance_km,
    eve_mode="none",
    eve_intensity=0.0,
    profile="iid",
    pns_strategy=None,
    rng=None,
    intensities=(MU_SIGNAL, MU_DECOY, MU_VACUUM),
    probs=DECOY_PROBS,
    mean_burst=2000,
    detector=None,
    **chan,
):
    """One BB84 run: weak coherent pulses, decoy intensities, real loss,
    dark counts, and a physically-implemented Eve.
    eve_mode in {'none','intercept_resend','pns','pns_matched','blocking','loss_manipulation','mixed_pns_ir'}  (mixed_pns_ir: item 32)
      'pns' blocks single-photon pulses (Y1 = 0): a non-adaptive attacker that the decoy states expose by construction.
      'pns_matched' (review B1) is the adaptive counterpart: Eve throttles the multi-photon pulses she splits so that Bob's click probability
      for EVERY photon number equals the honest channel's -- gain, QBER and all decoy statistics are unchanged, so nothing in the
      observable data can reveal it (privacy amplification, not detection, is the defence).
    detector (review B7/B8): optional dict of honest-hardware effects, all off by default --
      afterpulse   probability that a click triggers a spurious click (error 0.5) in the next pulse slot,
      dead_pulses  number of pulse slots a detector is blind after a click,
      drift_amp    relative amplitude of a slow sinusoidal drift of the detector error rate during the run (B8),
      drift_cycles number of drift periods per run.
    """
    rng = rng or _fallback_rng()
    det = dict(afterpulse=0.0, dead_pulses=0, drift_amp=0.0, drift_cycles=1.0)
    det.update(detector or {})
    mu_sig = intensities[0]
    ch = channel_model(distance_km, mu=mu_sig, **chan)
    eta, Y0, edet, e0 = ch["eta"], ch["Y0"], ch["e_detector"], ch["e_0"]

    k = rng.choice(len(intensities), size=N, p=probs)
    mu_i = np.asarray(intensities)[k]
    n = rng.poisson(mu_i)  # ALWAYS sampled (P2)
    bit_A = rng.integers(0, 2, N)
    bas_A = rng.integers(0, 2, N)
    bas_B = rng.integers(0, 2, N)

    active = attack_schedule(N, eve_intensity, profile, rng, mean_burst)  # P4
    eta_eff = np.full(N, float(eta))
    extra_err = np.zeros(N)
    if (
        det["drift_amp"] > 0
    ):  # B8: slow drift of the misalignment error (temperature / polarisation drift), a property of the LINK
        _ph = rng.random() * 2 * np.pi
        edet = np.clip(
            edet
            * (
                1.0
                + det["drift_amp"]
                * np.sin(2 * np.pi * det["drift_cycles"] * np.arange(N) / N + _ph)
            ),
            0.0,
            0.5,
        )

    if det.get("noise_profile") == "bursty":
        edet = np.clip(edet * honest_noise_multiplier(N, det, rng), 0, 0.5)

    # These two draws are made UNCONDITIONALLY (not just inside the
    # matching branch) so every eve_mode consumes the exact same amount of
    # the rng stream -- otherwise a zero-strength ('active' all False)
    # 'pns'/'intercept_resend' run would still land on a different point in
    # the random sequence than a 'none' run, which is an unnecessary,
    # avoidable confound for controls like the L-8 zero-strength audit
    # (P16) that compare classes at eve_intensity=0.
    bas_E = rng.integers(0, 2, N)
    pns_strat = pns_strategy or make_pns_strategy(distance_km, mu_sig, **chan)
    fwd = (n >= pns_strat["n_split"]) & (rng.random(N) < pns_strat["t_fwd"])

    if eve_mode == "intercept_resend":
        extra_err = np.where(active & (bas_E != bas_A), 0.5, 0.0)
    elif eve_mode == "pns":
        pass  # review fix B2: handled below through the click probability (eta_bob, n-1 photons), not eta_eff
    elif eve_mode == "blocking":
        eta_eff = np.where(active, 0.0, eta)
    elif eve_mode == "loss_manipulation":
        boost = 1.0 / max(1.0 - eve_intensity, 1e-6)
        # B6: even a lossless line cannot raise the click probability above Bob's detector efficiency eta_bob
        _eb = chan.get("eta_bob") if chan.get("eta_bob") is not None else GYS["eta_bob"]
        eta_eff = np.where(
            active, 0.0, min(eta * boost, _eb if BB84_LOSS_CAP_ETA_BOB else 1.0)
        )
    elif eve_mode == "mixed_pns_ir":
        # Draft 2, item 32: PNS and intercept-resend run TOGETHER on the
        # same active/intensity schedule. They write to DISJOINT outputs
        # (extra_err vs. eta_eff) so nothing here special-cases their
        # interaction -- this simple sum IS the composition, not an
        # approximation of it (see Section 22.3 for why that matters).
        extra_err = np.where(active & (bas_E != bas_A), 0.5, 0.0)

    p_sig = 1.0 - (1.0 - eta_eff) ** n
    if eve_mode in ("pns", "mixed_pns_ir"):
        # Review fix B2: a forwarded pulse reaches Bob with n-1 photons, each detected with prob. eta_bob.
        eta_bob_ = (
            chan.get("eta_bob") if chan.get("eta_bob") is not None else GYS["eta_bob"]
        )
        p_click_fwd = 1.0 - (1.0 - eta_bob_) ** np.maximum(n - 1, 0)
        p_sig = np.where(active, np.where(fwd, p_click_fwd, 0.0), p_sig)
    if eve_mode == "pns_matched":
        _eb2 = (
            chan.get("eta_bob") if chan.get("eta_bob") is not None else GYS["eta_bob"]
        )
        _honest = 1.0 - (1.0 - eta_eff) ** n
        _pns = 1.0 - (1.0 - _eb2) ** np.maximum(n - 1, 0)
        _t = np.where(
            _pns > 0, np.minimum(1.0, _honest / np.maximum(_pns, 1e-300)), 0.0
        )
        p_sig = np.where(
            active, np.where(n >= 2, _t * _pns, _honest), p_sig
        )  # == honest wherever Eve can throttle
    sig_cl = rng.random(N) < p_sig
    dark = rng.random(N) < Y0
    click = sig_cl | dark
    if (
        det["afterpulse"] > 0
    ):  # B7: afterpulse = spurious click after a click, independent of the pulse intensity (hits the vacuum yield too)
        _prev = np.concatenate([[False], click[:-1]])
        click = click | (_prev & (rng.random(N) < det["afterpulse"]))
    if (
        det["dead_pulses"] > 0
    ):  # B7: dead time -- clicks inside the blind window after a click are lost
        _idx = np.flatnonzero(click)
        _last = -(10**9)
        _dead = int(det["dead_pulses"])
        for _i in _idx:
            if _i - _last <= _dead:
                click[_i] = False
                sig_cl[_i] = False
            else:
                _last = _i

    p_err = np.where(sig_cl, np.clip(edet + extra_err * (1 - 2 * edet), 0, 1), e0)
    bit_B = np.where(rng.random(N) < p_err, 1 - bit_A, bit_A)
    sift = click & (bas_A == bas_B)

    Q, E, counts = {}, {}, {}
    for j, m in enumerate(intensities):
        sel = k == j
        s2 = sel & sift
        counts[m] = int(sel.sum())
        Q[m] = float(click[sel].mean()) if sel.any() else 0.0
        E[m] = float((bit_A[s2] != bit_B[s2]).mean()) if s2.any() else e0

    return dict(
        bit_A=bit_A,
        bit_B=bit_B,
        bas_A=bas_A,
        bas_B=bas_B,
        n=n,
        k=k,
        click=click,
        sift=sift,
        N=N,
        Q=Q,
        E=E,
        counts=counts,
        intensities=list(intensities),
        theory=ch,
    )


def simulate_bkm07_batch(
    N,
    distance_km,
    eve_mode,
    eve_fwd,
    eve_ret,
    rng=None,
    p_prep=None,
    p_meas=None,
    p_ret=None,
    detector=None,
    ctrl_fwd_noise=None,
    **chan,
):
    """Vectorised BKM07 simulator (Draft 2, item 7) -- see the callout above
    for why this is physically identical to simulate_bkm07_pulse. Draws
    more rng values per pulse than the scalar loop (it evaluates every
    branch for every row, then masks), so it does NOT reproduce the scalar
    version bit-for-bit at a shared seed -- only statistically, which is
    all downstream code needs (same convention as simulate_bb84_decoy vs.
    the retired per-pulse simulate_bb84_pulse)."""
    rng = rng or _fallback_rng()
    ch = channel_model(distance_km, **chan)
    eta = ch["eta"]
    p_prep = ch["e_detector"] if p_prep is None else p_prep
    # Review fix B7: forward measurement, re-preparation and return-leg errors are separate parameters
    # (all default to e_detector, which reproduces the previous model exactly).
    p_meas = ch["e_detector"] if p_meas is None else p_meas
    p_ret = ch["e_detector"] if p_ret is None else p_ret
    ctrl_fwd_noise = (
        BKM07_CTRL_FORWARD_NOISE if ctrl_fwd_noise is None else ctrl_fwd_noise
    )
    det = dict(drift_amp=0.0, drift_cycles=1.0)
    det.update(detector or {})
    if (
        det["drift_amp"] > 0
    ):  # B8: slow drift of the honest error rate, common to all error sources of the link
        _f = 1.0 + det["drift_amp"] * np.sin(
            2 * np.pi * det["drift_cycles"] * np.arange(N) / N
            + rng.random() * 2 * np.pi
        )
        p_prep, p_meas, p_ret = (
            np.clip(np.asarray(v) * _f, 0.0, 0.5) for v in (p_prep, p_meas, p_ret)
        )

    if det.get("noise_profile") == "bursty":
        _burst = honest_noise_multiplier(N, det, rng)
        p_prep, p_meas, p_ret = (
            np.clip(np.asarray(v) * _burst, 0, 0.5) for v in (p_prep, p_meas, p_ret)
        )

    bit_A = rng.integers(0, 2, N)
    basis_A = rng.integers(0, 2, N)
    cur_bit, cur_basis = (
        bit_A.copy(),
        basis_A.copy(),
    )  # the travelling carrier's (bit, basis)

    alive = rng.random(N) < eta  # survived the forward leg

    if eve_mode != "none":
        eve_fwd_hit = alive & (rng.random(N) < eve_fwd)
        basis_Ef = rng.integers(0, 2, N)
        same = basis_Ef == cur_basis
        meas_bit = np.where(same, cur_bit, rng.integers(0, 2, N))
        cur_bit = np.where(eve_fwd_hit, meas_bit, cur_bit)
        cur_basis = np.where(eve_fwd_hit, basis_Ef, cur_basis)

    bob_sift = alive & (rng.random(N) < 0.5)
    bob_ctrl = alive & ~bob_sift

    noisy_B = rng.random(N) < (2 * p_meas)
    same0 = cur_basis == 0
    meas_B = np.where(same0, cur_bit, rng.integers(0, 2, N))
    meas_B = np.where(noisy_B, rng.integers(0, 2, N), meas_B)
    bit_B = np.where(bob_sift, meas_B, 0)  # Bob's retained classical SIFT bit
    flip_prep = rng.random(N) < p_prep
    sent_bit = np.where(bob_sift & flip_prep, 1 - bit_B, bit_B)
    cur_bit = np.where(bob_sift, sent_bit, cur_bit)
    cur_basis = np.where(bob_sift, 0, cur_basis)  # ... re-prepared in Z
    if ctrl_fwd_noise:
        # B7/B3 physics fix: the forward-leg misalignment acts on the photon BEFORE Bob decides what to do with it. For SIFT rounds it is
        # Bob's measurement noise (p_meas); CTRL rounds used to skip it, so their honest error was ONE noisy step instead of two.
        noisy_fwd_ctrl = bob_ctrl & (rng.random(N) < (2 * p_meas))
        cur_bit = np.where(noisy_fwd_ctrl, rng.integers(0, 2, N), cur_bit)

    if eve_mode != "none":
        eve_ret_hit = alive & (rng.random(N) < eve_ret)
        basis_Er = rng.integers(0, 2, N)
        same_r = basis_Er == cur_basis
        meas_bit_r = np.where(same_r, cur_bit, rng.integers(0, 2, N))
        cur_bit = np.where(eve_ret_hit, meas_bit_r, cur_bit)
        cur_basis = np.where(eve_ret_hit, basis_Er, cur_basis)

    survived = alive & (rng.random(N) < eta)  # survived the return leg too (t^2, not t)

    meas_basis_final = np.where(
        bob_ctrl, basis_A, 0
    )  # CTRL: Alice measures in her OWN basis_A
    noisy_final = rng.random(N) < (2 * p_ret)
    same_f = cur_basis == meas_basis_final
    meas_final = np.where(same_f, cur_bit, rng.integers(0, 2, N))
    bit_A_final = np.where(noisy_final, rng.integers(0, 2, N), meas_final)

    round_type = np.full(N, "", dtype="<U12")
    round_type[bob_ctrl & (basis_A == 0)] = "CTRL_Z"
    round_type[bob_ctrl & (basis_A == 1)] = "CTRL_X"
    round_type[bob_sift & (basis_A == 0)] = "SIFT_KEY"
    round_type[bob_sift & (basis_A == 1)] = "SIFT_MONITOR"

    return dict(
        lost=~survived,
        survived=survived,
        bit_A=bit_A,
        basis_A=basis_A,
        bob_mode=np.where(bob_sift, "SIFT", "CTRL"),
        round_type=round_type,
        bit_B=bit_B,
        bit_A_final=bit_A_final,
    )


def run_e91(
    n_pairs,
    V=0.95,
    eve_mode="none",
    eve_intensity=0.0,
    lam=0.3,
    profile="iid",
    rng=None,
    mean_burst=2000,
    drift_amp=0.0,
    drift_cycles=1.0,
    n_drift_segments=32,
    noise_profile="stationary",
    burst_multiplier=3.0,
):
    """One E91 run. Outcomes are drawn from the exact Born-rule joint distribution for each (setting pair, channel condition) --
    multinomial sampling from the Born probabilities is identically distributed to shot-by-shot simulation.

    drift_amp (review B8): the honest noise (1 - V) drifts slowly, (1 - V_t) = (1 - V)(1 + drift_amp sin(2 pi cycles t / n + phase)),
    piecewise constant over n_drift_segments segments; 0 reproduces the earlier single-visibility behaviour exactly."""
    rng = rng or _fallback_rng()
    attacked = attack_schedule(n_pairs, eve_intensity, profile, rng, mean_burst)

    a_keys = list(ALICE_ANGLES.keys())
    b_keys = list(BOB_ANGLES.keys())
    ak = rng.choice(a_keys, size=n_pairs)
    bk = rng.choice(b_keys, size=n_pairs)
    ra = np.zeros(n_pairs, dtype=int)
    rb = np.zeros(n_pairs, dtype=int)

    if noise_profile == "bursty":
        edges = np.linspace(0, n_pairs, n_drift_segments + 1, dtype=int)
        _f = honest_noise_multiplier(
            n_drift_segments,
            dict(
                noise_profile="bursty", burst_multiplier=burst_multiplier, burst_block=1
            ),
            rng,
        )
        Vs = np.clip(1 - (1 - V) * _f, 0, 1)
    elif drift_amp > 0:
        edges = np.linspace(0, n_pairs, n_drift_segments + 1, dtype=int)
        _ph = rng.random() * 2 * np.pi
        mids = 0.5 * (edges[:-1] + edges[1:]) / max(n_pairs, 1)
        Vs = np.clip(
            1.0
            - (1.0 - V)
            * (1.0 + drift_amp * np.sin(2 * np.pi * drift_cycles * mids + _ph)),
            0.0,
            1.0,
        )
    else:
        edges, Vs = np.array([0, n_pairs]), np.array([V])

    for lo, hi, V_seg in zip(edges[:-1], edges[1:], Vs):
        rho0 = werner_state(float(V_seg))
        if eve_mode == "intercept_resend":
            rho_att = eve_ir_channel(rho0)
        elif eve_mode == "ancilla":
            rho_att = eve_ancilla_bob(rho0, lam)
        elif eve_mode in (
            "extra_depolarisation",
            "loss_manipulation",
        ):  # review fix B5: renamed -- it is white noise, not a loss attack
            rho_att = eve_loss_manipulation(rho0, delta=min(0.35, 0.5 * eve_intensity))
        else:
            rho_att = rho0
        seg = slice(lo, hi)
        for a_name, a_deg in ALICE_ANGLES.items():
            for b_name, b_deg in BOB_ANGLES.items():
                for att in (False, True):
                    m = np.zeros(n_pairs, dtype=bool)
                    m[seg] = (
                        (ak[seg] == a_name)
                        & (bk[seg] == b_name)
                        & (attacked[seg] == att)
                    )
                    cnt = int(m.sum())
                    if cnt == 0:
                        continue
                    p = joint_probs(rho_att if att else rho0, a_deg, b_deg)
                    keys = list(p)
                    idx = rng.choice(len(keys), size=cnt, p=[p[kk] for kk in keys])
                    outs = np.array(keys)[idx]
                    ra[m], rb[m] = outs[:, 0], outs[:, 1]

    return ak, bk, ra, rb


def window_qber(a_choice, b_choice, r_a, r_b, start, end):
    key_a, key_b = [], []
    for ac, bc in KEY_PAIRS:
        m = (a_choice[start:end] == ac) & (b_choice[start:end] == bc)
        key_a.extend(r_a[start:end][m])
        key_b.extend(-r_b[start:end][m])  # flip: singlet -> anti-correlated
    if len(key_a) == 0:
        return np.nan
    key_a, key_b = np.array(key_a), np.array(key_b)
    return np.mean(key_a != key_b)


def window_chsh(a_choice, b_choice, r_a, r_b, start, end):
    corrs = []
    for ac, bc in CHSH_PAIRS:
        m = (a_choice[start:end] == ac) & (b_choice[start:end] == bc)
        if m.sum() == 0:
            return np.nan
        corrs.append(np.mean(r_a[start:end][m] * r_b[start:end][m]))
    return sum(s * c for s, c in zip(CHSH_SIGNS, corrs))


def windowed_traces(a_choice, b_choice, r_a, r_b, n_windows):
    n = len(a_choice)
    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    qber_trace, chsh_trace = [], []
    for start, end in zip(edges[:-1], edges[1:]):
        qber_trace.append(window_qber(a_choice, b_choice, r_a, r_b, start, end))
        chsh_trace.append(window_chsh(a_choice, b_choice, r_a, r_b, start, end))
    return np.array(qber_trace), np.array(chsh_trace)


def spectral_entropy(x):
    return _spectral_entropy(x)


def autocorr_lag1(x):
    return _autocorr_lag1(x)


def binary_entropy(p):
    return _binary_entropy(p)


def _binary_entropy(p):
    """H(p) = -p log2(p) - (1-p) log2(1-p)."""
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _spectral_entropy(x, drop_dc=True):
    """Normalised Shannon entropy of the power spectrum, in [0, 1].

    RELIABILITY NOTE (Draft 2, item 4 -- doc/code mismatch fixed): with W
    windows the rfft yields floor(W/2) usable bins after dropping DC. The
    plug-in entropy estimator is strongly negatively biased for small bin
    counts (Paninski, Neural Comput. 15, 1191 (2003)). Draft 1's comment
    HERE claimed this only returns a value for W >= 64 and NaN otherwise --
    that was never what the code below does: it only returns NaN when
    fewer than 8 non-NaN window values are available (see the `len(x) < 8`
    check), which is why E91's 32-window traces (Section 3.4) got a real
    number, not NaN, despite sitting below the claimed W>=64 cutoff.
    The actual behaviour is intentional and is kept as-is here (raising
    E91's window count to remove the bias is a scale decision made
    separately for Section 5/Section 19 -- see item 8); what changed is
    only this comment, so it now says what the code actually guarantees:
    a real (but increasingly negatively-biased-as-W-shrinks) value for any
    trace with >= 8 usable windows, NaN below that. Read this feature's
    ABSOLUTE value with that bias in mind, especially for W < 64; relative
    comparisons (honest vs. attacked at the SAME W) remain valid since the
    bias is systematic and hits both classes equally at a given W.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return np.nan
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if drop_dc:
        spec = spec[1:]
    tot = spec.sum()
    if tot <= 0:
        return 0.0
    p = spec / tot
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


def _autocorr_lag1(x):
    """Lag-1 autocorrelation. Near 0 = uncorrelated; positive = clustering."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    x0, x1 = x[:-1] - np.mean(x[:-1]), x[1:] - np.mean(x[1:])
    denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    return 0.0 if denom == 0 else float(np.sum(x0 * x1) / denom)


def _dispersion_index(window_errors, window_counts):
    """Ratio of observed per-window QBER variance to the binomial floor.

    D = Var(Qhat) * nbar / (Qbar (1 - Qbar))

    D=1 : the QBER trace is consistent with a STATIONARY process observed
          through finite-sample noise (honest channel, or i.i.d. Eve).
    D>1 : over-dispersion -- the underlying error rate itself varies in
          time (bursty/drifting Eve, or a drifting channel).

    Replaces the raw `qber_variance`, which is a deterministic function of
    the mean QBER (measured ratio to this floor: 0.90-0.98) and therefore
    duplicates `qber_total` rather than adding information.
    """
    we = np.asarray(window_errors, dtype=float)
    wc = np.asarray(window_counts, dtype=float)
    ok = wc > 0
    if ok.sum() < 4:
        return np.nan
    q = we[ok] / wc[ok]
    qb = we[ok].sum() / wc[ok].sum()
    nb = wc[ok].mean()
    floor = qb * (1 - qb) / nb
    return float(np.var(q, ddof=1) / floor) if floor > 0 else np.nan


def _jump_energy_rel(x):
    """Review fix E6: normalised jump energy for traces without a binomial floor (E91's CHSH trace):
    sum of squared consecutive differences / its expectation for a white sequence with the same variance (~1 if stationary)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    v = np.var(x, ddof=1)
    return float(np.sum(np.diff(x) ** 2) / (2 * (len(x) - 1) * v)) if v > 0 else np.nan


def _z_vs_baseline(q, K, baseline):
    """Review fix B8: z-score of the run's error rate against a per-link attack-free commissioning run
    (q0 from K0 key bits): z = (q - q0) / sqrt(q0(1-q0)(1/K0 + 1/K)). 0 when no baseline is supplied."""
    if baseline is None or not np.isfinite(q):
        return 0.0
    q0 = float(np.clip(baseline["q0"], 1e-4, 0.5))
    K0 = max(int(baseline["K0"]), 1)
    return float((q - q0) / np.sqrt(q0 * (1 - q0) * (1.0 / K0 + 1.0 / max(int(K), 1))))


def _jump_energy_norm(window_errors, window_counts):
    """Sum of squared consecutive QBER differences, normalised by the value
    expected for a white (stationary) sequence: 2*(W-1)*Qbar(1-Qbar)/nbar.
    ~1 under stationarity, >1 for abrupt transitions."""
    we = np.asarray(window_errors, dtype=float)
    wc = np.asarray(window_counts, dtype=float)
    ok = wc > 0
    if ok.sum() < 3:
        return np.nan
    q = we[ok] / wc[ok]
    qb = we[ok].sum() / wc[ok].sum()
    nb = wc[ok].mean()
    expected = 2 * (len(q) - 1) * qb * (1 - qb) / nb
    return float(np.sum(np.diff(q) ** 2) / expected) if expected > 0 else np.nan


def decoy_estimate(Q_mu, E_mu, Q_nu, E_nu, Y0_obs, mu=MU_SIGNAL, nu=MU_DECOY, e0=0.5):
    """Vacuum+Weak decoy estimation of the single-photon yield and error rate."""
    Y1L = (
        mu
        / (mu * nu - nu**2)
        * (
            Q_nu * np.exp(nu)
            - Q_mu * np.exp(mu) * nu**2 / mu**2
            - (mu**2 - nu**2) / mu**2 * Y0_obs
        )
    )
    Y1L = max(float(Y1L), 0.0)
    Q1L = Y1L * mu * np.exp(-mu)
    e1U = ((E_nu * Q_nu * np.exp(nu) - e0 * Y0_obs) / (Y1L * nu)) if Y1L > 0 else 0.5
    return Y1L, Q1L, float(np.clip(e1U, 0.0, 0.5))


def secure_key_rate(Q_mu, E_mu, Q1, e1, f_ec=None, q=0.5):
    """GLLP + decoy asymptotic key rate; Lo, Ma & Chen PRL 94, 230504 (2005) Eq. 1.
    ASYMPTOTIC rate, used here as a channel-state FEATURE, not a security
    guarantee -- finite-key composable security needs additional terms
    (Tomamichel et al. 2012) not computed here."""
    f_ec = GYS["f_ec"] if f_ec is None else f_ec
    return max(
        q * (-Q_mu * f_ec * _binary_entropy(E_mu) + Q1 * (1.0 - _binary_entropy(e1))),
        0.0,
    )


BB84_BASE_FEATURES = [
    "qber_total",
    "qber_z",
    "qber_x",
    "gain_mu",
    "sifted_rate",
    "qber_dispersion",
    "jump_energy_norm",
    "spectral_entropy",
    "autocorr_lag1",
    "y1_lower",
    "e1_upper",
    "r_secure",
    "gain_ratio_nu_mu",
    "h_qber",
]


BB84_FEATURE_NAMES = BB84_BASE_FEATURES + ["gain_vs_expected", "z_qber"]


def _truncate_bb84_run(run, K):
    """Review fix C3: cut a run at the pulse where the K-th sifted SIGNAL bit occurs, so every run carries exactly K
    key bits (equal information) instead of 1.3K +- noise; recomputes the per-intensity gains/QBERs."""
    s = run["sift"] & (run["k"] == 0)
    cs = np.cumsum(s)
    if len(cs) == 0 or cs[-1] < K:
        return run, int(run["N"])
    cut = int(np.searchsorted(cs, K)) + 1
    r = dict(run)
    N0 = run["N"]
    for key in ("bit_A", "bit_B", "bas_A", "bas_B", "n", "k", "click", "sift"):
        r[key] = run[key][:cut]
    e0 = run["theory"]["e_0"]
    Q, E, counts = {}, {}, {}
    for j, m in enumerate(run["intensities"]):
        sel = r["k"] == j
        s2 = sel & r["sift"]
        counts[m] = int(sel.sum())
        Q[m] = float(r["click"][sel].mean()) if sel.any() else 0.0
        E[m] = float((r["bit_A"][s2] != r["bit_B"][s2]).mean()) if s2.any() else e0
    r.update(N=cut, Q=Q, E=E, counts=counts)
    return r, cut


def _truncate_bkm07(b, K):
    """Review fix C3 for BKM07: cut at the K-th SURVIVED SIFT_KEY round."""
    key = b["survived"] & (b["round_type"] == "SIFT_KEY")
    cs = np.cumsum(key)
    if len(cs) == 0 or cs[-1] < K:
        return b, len(key)
    cut = int(np.searchsorted(cs, K)) + 1
    return {
        k: (v[:cut] if isinstance(v, np.ndarray) and v.shape[:1] == (len(key),) else v)
        for k, v in b.items()
    }, cut


def collect_bb84_features(
    N=None,
    distance_km=0.0,
    eve_mode="none",
    eve_intensity=0.0,
    profile="iid",
    pns_strategy=None,
    n_windows=64,
    rng=None,
    target_k_signal_bits=None,
    k_margin=K_MARGIN,
    max_pulses=20_000_000,
    mean_burst=None,
    baseline=None,
    truncate=True,
    detector=None,
    return_record=False,
    **chan,
):
    """Run N BB84 pulses through the physical channel (P1/P2) and compress
    into the BB84_FEATURE_NAMES feature vector, via the vectorised
    decoy-state simulator (P5).

    Draft 2, item 8: pass `target_k_signal_bits=K` instead of `N` to size N
    automatically from the closed-form channel model so the run nets
    approximately K sifted SIGNAL bits regardless of distance -- the
    equal-information sampling rule used by Section 5's main datasets and
    Section 19's rebuilt benchmark. N is still accepted directly for every
    other caller (Section 16's N x W sensitivity study, Section 4's
    calibration checks, etc.), which keep Draft 1's fixed-N behaviour
    unchanged. Either way the returned dict now also carries the ACTUALLY
    achieved counts under underscore-prefixed keys -- '_k_achieved' /
    '_N_used' -- which are never treated as physics features
    (BB84_FEATURE_NAMES doesn't list them) but feed the leakage audit's
    real N_per_run and the AUC-vs-key-rounds plot (item 6).
    """
    rng = rng or _fallback_rng()
    N_capped = False
    if target_k_signal_bits is not None:
        _ch_est = channel_model(distance_km, **chan)
        _rate = (
            DECOY_PROBS[0] * _ch_est["gain"] * 0.5
        )  # signal share x gain x basis-match prob
        _N_wanted = int(np.ceil(target_k_signal_bits / max(_rate, 1e-15) * k_margin))
        N, N_capped = (
            min(_N_wanted, max_pulses),
            _N_wanted > max_pulses,
        )  # Draft 2.1: cap is now REPORTED
    elif N is None:
        raise ValueError("collect_bb84_features needs either N or target_k_signal_bits")

    # Final draft: burst length scales with the run (~1/16 of it). The fixed 2,000-pulse default was
    # shorter than one analysis window by orders of magnitude at N ~ 1e6-2e7, so 'bursty' attacks were
    # statistically iid and the temporal features had nothing to detect.
    # B9: the burst length is no longer one fixed fraction of the run -- it is drawn per run from {N/8, N/16, N/32, N/64}
    _mb = (
        mean_burst
        if mean_burst is not None
        else max(2000, int(N) // int(rng.choice([8, 16, 32, 64])))
    )
    run = simulate_bb84_decoy(
        N,
        distance_km,
        eve_mode,
        eve_intensity,
        profile=profile,
        pns_strategy=pns_strategy,
        rng=rng,
        mean_burst=_mb,
        detector=detector,
        **chan,
    )

    run = _extend_to_k(
        run,
        lambda n: simulate_bb84_decoy(
            n,
            distance_km,
            eve_mode,
            eve_intensity,
            profile=profile,
            pns_strategy=pns_strategy,
            rng=rng,
            mean_burst=_mb,
            detector=detector,
            **chan,
        ),
        lambda r: r["sift"] & (r["k"] == 0),
        target_k_signal_bits,
        N,
    )
    N_gen = int(run["N"])
    N = N_gen
    if truncate and target_k_signal_bits is not None:
        run, N = _truncate_bb84_run(run, target_k_signal_bits)
    bit_A, bit_B = run["bit_A"], run["bit_B"]
    bas_A, sift = run["bas_A"], run["sift"]
    is_sig = run["k"] == 0  # signal-intensity pulses only
    s = sift & is_sig
    sz = s & (bas_A == 0)
    sx = s & (bas_A == 1)

    qber_total = float((bit_A[s] != bit_B[s]).mean()) if s.any() else 0.0
    qber_z = float((bit_A[sz] != bit_B[sz]).mean()) if sz.any() else 0.0
    qber_x = float((bit_A[sx] != bit_B[sx]).mean()) if sx.any() else 0.0
    gain_mu = run["Q"][run["intensities"][0]]
    sifted_rate = float(s.sum() / max(is_sig.sum(), 1))

    # window loop keeps (error_count, sifted_count) per window, not a
    # bare ratio -- the dispersion index needs the counts to compute the
    # binomial floor. Parameterised by NUMBER OF WINDOWS, not pulses per
    # window, so a fixed window count gives comparable estimator variance
    # regardless of distance/loss.
    edges = np.linspace(0, N, n_windows + 1, dtype=int)
    w_err, w_cnt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = s[lo:hi]
        w_cnt.append(int(m.sum()))
        w_err.append(int((bit_A[lo:hi][m] != bit_B[lo:hi][m]).sum()))
    w_err, w_cnt = np.array(w_err), np.array(w_cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        wq = np.where(w_cnt > 0, w_err / np.maximum(w_cnt, 1), np.nan)

    # Decoy-state block (P5's estimators)
    mu_s, mu_d, mu_v = run["intensities"]
    Y0_obs = run["Q"][mu_v]
    y1, q1, e1 = decoy_estimate(
        run["Q"][mu_s],
        run["E"][mu_s],
        run["Q"][mu_d],
        run["E"][mu_d],
        Y0_obs,
        mu=mu_s,
        nu=mu_d,
    )
    r_sec = secure_key_rate(run["Q"][mu_s], run["E"][mu_s], q1, e1)
    ratio = float(
        (run["Q"][mu_d] * np.exp(mu_d)) / max(run["Q"][mu_s] * np.exp(mu_s), 1e-18)
    )

    features = {
        "qber_total": qber_total,
        "qber_z": qber_z,
        "qber_x": qber_x,
        "gain_mu": float(gain_mu),
        "sifted_rate": sifted_rate,
        "qber_dispersion": _dispersion_index(w_err, w_cnt),
        "jump_energy_norm": _jump_energy_norm(w_err, w_cnt),
        "spectral_entropy": _spectral_entropy(wq),
        "autocorr_lag1": _autocorr_lag1(wq[~np.isnan(wq)]),
        "y1_lower": float(y1),
        "e1_upper": float(e1),
        "r_secure": float(r_sec),
        "gain_ratio_nu_mu": ratio,
        "h_qber": float(_binary_entropy(qber_total)),
        # review fix B6: link-budget expectation as an input -- observed gain / gain expected at this distance, so a lower
        # click rate caused by blocking is distinguishable from a longer fibre.
        "gain_vs_expected": float(
            gain_mu / max(channel_model(distance_km, **chan)["gain"], 1e-18)
        ),
        # review fix B8: z-score against the per-link commissioning baseline.
        "z_qber": _z_vs_baseline(qber_total, int(s.sum()), baseline),
        "_k_achieved": int(
            s.sum()
        ),  # Draft 2, item 8/6: sifted signal bits actually obtained
        "_N_used": int(N),
        "_N_trunc": int(N),
        "_N_capped": bool(
            N_capped
        ),  # initial draw hit its batch limit; continuation still supplies exact K
    }

    features["_N_generated"] = N_gen
    features["_key_yield"] = features["_k_achieved"] / max(int(N), 1)
    features["_detection_yield"] = float(run["click"].mean())
    features["_loss_rate"] = 1.0 - features["_detection_yield"]
    if truncate and target_k_signal_bits is not None:
        assert features["_k_achieved"] == target_k_signal_bits
    if return_record:
        features["_record"] = run
    return features


def collect_bkm07_features(
    N=None,
    distance_km=0.0,
    eve_mode="none",
    eve_fwd=0.0,
    eve_ret=0.0,
    n_windows=64,
    rng=None,
    target_k_key_rounds=None,
    k_margin=K_MARGIN,
    max_pulses=15_000_000,
    baseline=None,
    truncate=True,
    detector=None,
    return_record=False,
    **chan,
):
    """Run N BKM07 round trips and compress into a feature vector.

    Alice measures SIFT returns in her OWN preparation basis
    (not always Z); X-SIFT rounds are protocol-discarded and kept only as
    an auxiliary `ret_monitor` (renamed from `qber_xsr`, which the previous
    version reported as if it were a protocol observable -- it was not:
    Bob re-prepares in Z, Alice measured in Z, so it measured only
    return-leg channel noise); the channel is applied TWICE (round trip:
    t^2, not t); `xctrl_variance` replaced by a dispersion index (P6/P11).

    Draft 2, item 7: rewritten to call the vectorised `simulate_bkm07_batch`
    instead of looping `simulate_bkm07_pulse` N times in pure Python --
    same physics, same feature definitions, just array arithmetic instead
    of a per-pulse loop (see the callout above `simulate_bkm07_batch`).

    Draft 2, item 8: pass `target_k_key_rounds=K` instead of `N` to size N
    from (round-trip survival)^2 x (SIFT_KEY share of survivors) so the run
    nets approximately K SIFT_KEY rounds regardless of distance -- now
    affordable at the K this notebook uses (2,000) precisely because of the
    item-7 vectorisation above. Returns '_k_achieved' / '_N_used' the same
    way collect_bb84_features does (see its docstring).
    """
    rng = rng or _fallback_rng()
    N_capped = False
    if target_k_key_rounds is not None:
        _ch_est = channel_model(distance_km, **chan)
        _rate = (_ch_est["eta"] ** 2) * 0.25  # round-trip survival x SIFT_KEY share
        _N_wanted = int(np.ceil(target_k_key_rounds / max(_rate, 1e-15) * k_margin))
        N, N_capped = (
            min(_N_wanted, max_pulses),
            _N_wanted > max_pulses,
        )  # Draft 2.1: cap is now REPORTED
    elif N is None:
        raise ValueError("collect_bkm07_features needs either N or target_k_key_rounds")

    b = simulate_bkm07_batch(
        N, distance_km, eve_mode, eve_fwd, eve_ret, rng=rng, detector=detector, **chan
    )
    b = _extend_to_k(
        b,
        lambda n: simulate_bkm07_batch(
            n,
            distance_km,
            eve_mode,
            eve_fwd,
            eve_ret,
            rng=rng,
            detector=detector,
            **chan,
        ),
        lambda r: r["survived"] & (r["round_type"] == "SIFT_KEY"),
        target_k_key_rounds,
        N,
    )
    N_gen = len(b["survived"])
    N = N_gen
    if truncate and target_k_key_rounds is not None:
        b, N = _truncate_bkm07(b, target_k_key_rounds)
    surv = b["survived"]
    rt = b["round_type"]
    bit_A, bit_B, bit_Af = b["bit_A"], b["bit_B"], b["bit_A_final"]

    z_sft = surv & (rt == "SIFT_KEY")
    x_sft = surv & (rt == "SIFT_MONITOR")
    z_ctrl = surv & (rt == "CTRL_Z")
    x_ctrl = surv & (rt == "CTRL_X")
    z_sft_t, x_sft_t = int(z_sft.sum()), int(x_sft.sum())
    z_ctrl_t, x_ctrl_t = int(z_ctrl.sum()), int(x_ctrl.sum())

    qber_zs = float((bit_A[z_sft] != bit_B[z_sft]).mean()) if z_sft_t else 0.0
    # SIFT key agreement is between Bob's retained measurement and Alice's
    # returned measurement. Alice's original preparation is a separate monitor.
    qber_key = float((bit_B[z_sft] != bit_Af[z_sft]).mean()) if z_sft_t else 0.0
    qber_zsr = qber_key  # compatibility alias; do not count as independent evidence
    ret_monitor = (
        float((bit_B[x_sft] != bit_Af[x_sft]).mean()) if x_sft_t else 0.0
    )  # was qber_xsr
    qber_zc = float((bit_A[z_ctrl] != bit_Af[z_ctrl]).mean()) if z_ctrl_t else 0.0
    qber_xc = float((bit_A[x_ctrl] != bit_Af[x_ctrl]).mean()) if x_ctrl_t else 0.0

    asymmetry = abs(qber_zsr - qber_zs)
    ctrl_sift_ratio = float((qber_zc + qber_xc) / 2.0) / max(qber_zs, 1e-9)

    # Windowed CTRL_X trace, binned by POSITION AMONG SURVIVORS (boolean
    # indexing preserves original pulse order, matching the scalar
    # version's temporal ordering of its filtered `pulses` list).
    N_eff = max(int(surv.sum()), 1)
    rt_s, err_s = rt[surv], (bit_A[surv] != bit_Af[surv])
    is_ctrl_x = rt_s == "CTRL_X"
    err_ctrl_x = err_s & is_ctrl_x
    edges = np.linspace(0, N_eff, n_windows + 1, dtype=int)
    w_err, w_cnt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w_cnt.append(int(is_ctrl_x[lo:hi].sum()))
        w_err.append(int(err_ctrl_x[lo:hi].sum()))
    w_err, w_cnt = np.array(w_err), np.array(w_cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        wq = np.where(w_cnt > 0, w_err / np.maximum(w_cnt, 1), np.nan)

    qber_ctrl_avg = (qber_zc + qber_xc) / 2.0

    features = {
        "qber_key": qber_key,
        "qber_zs": qber_zs,
        "qber_zsr": qber_zsr,
        "ret_monitor": ret_monitor,  # was 'qber_xsr' (P10 rename)
        "qber_zc": qber_zc,
        "qber_xc": qber_xc,
        "asymmetry": asymmetry,
        "ctrl_sift_ratio": ctrl_sift_ratio,
        "xctrl_dispersion": _dispersion_index(w_err, w_cnt),
        "jump_energy": _jump_energy_norm(
            w_err, w_cnt
        ),  # review fix E6: normalised, same statistic as BB84
        "spectral_entropy": _spectral_entropy(wq),
        "autocorr_lag1": _autocorr_lag1(wq[~np.isnan(wq)]),
        "h_ctrl": float(_binary_entropy(qber_ctrl_avg)),  # was 'holevo_ie'
        "sifted_rate": z_sft_t / N_eff,
        "z_qber_ctrl": _z_vs_baseline(
            qber_ctrl_avg, z_ctrl_t + x_ctrl_t, baseline
        ),  # review fix B8
        "_qber_ctrl_avg": float(qber_ctrl_avg),
        "_n_ctrl": int(z_ctrl_t + x_ctrl_t),
        "_k_achieved": int(
            z_sft_t
        ),  # Draft 2, item 8/6: SIFT_KEY rounds actually obtained
        "_N_used": int(N),
        "_N_trunc": int(N),
        "_N_capped": bool(
            N_capped
        ),  # initial draw hit its batch limit; continuation still supplies exact K
    }

    features["_N_generated"] = N_gen
    features["_key_yield"] = features["_k_achieved"] / max(int(N), 1)
    features["_detection_yield"] = float(b["survived"].mean())
    features["_loss_rate"] = 1.0 - features["_detection_yield"]
    if truncate and target_k_key_rounds is not None:
        assert features["_k_achieved"] == target_k_key_rounds
    if return_record:
        features["_record"] = b
    return features


BKM_FEATURE_NAMES = [
    "qber_key",
    "qber_zs",
    "qber_zsr",
    "ret_monitor",
    "qber_zc",
    "qber_xc",
    "asymmetry",
    "ctrl_sift_ratio",
    "xctrl_dispersion",
    "jump_energy",
    "spectral_entropy",
    "autocorr_lag1",
    "h_ctrl",
    "sifted_rate",
    "z_qber_ctrl",
]


def e91_di_secure_key_rate(Q, S):
    """CHSH-based device-independent secure key rate lower bound
    (Acin, Brunner, Gisin, Massar, Pironio & Scarani, PRL 98, 230501,
    2007): r >= 1 - h(Q) - h( [1 + sqrt((S/2)^2 - 1)] / 2 ), where Q is
    the key-basis QBER and S the CHSH value.

    Requires |S| > 2 for the sqrt term to be real -- below that the
    correlations are compatible with a purely classical (local hidden
    variable) model and NO device-independent key is possible at all
    (Eve could in principle know everything), so the rate is clamped to
    0 rather than returning a complex/undefined value.

    ASYMPTOTIC rate, same caveat as secure_key_rate() in Section 3.2: a
    channel-state FEATURE here (any attack that lowers |S| or raises Q
    lowers it), not a finite-key composable security bound -- real
    device-independent QKD needs substantially more machinery (see
    Pironio et al., New J. Phys. 11, 045021, 2009, for the finite-key
    treatment)."""
    S = abs(S)
    if not np.isfinite(Q) or not np.isfinite(S) or S <= 2.0:
        return 0.0
    inner = (S / 2.0) ** 2 - 1.0
    if inner < 0:
        return 0.0
    p_err_eve = (1.0 + np.sqrt(inner)) / 2.0
    r = 1.0 - _binary_entropy(Q) - _binary_entropy(p_err_eve)
    return float(max(r, 0.0))


def extract_e91_features(
    n_pulses=2000,
    eve_mode="none",
    n_windows=20,
    rng=None,
    eve_intensity=0.0,
    lam=0.3,
    profile="iid",
    V=None,
    target_k_key_pairs=None,
    k_margin=K_MARGIN,
    baseline=None,
    truncate=True,
    detector=None,
    return_record=False,
):
    """Draft 2, item 8: pass `target_k_key_pairs=K` instead of `n_pulses` to
    size n_pulses so the run nets approximately K key pairs (settings
    a2-b1/a3-b2, which occur with probability 2/9 under the uniform random
    setting choice) -- the same equal-information rule as
    collect_bb84_features / collect_bkm07_features. n_pulses is still
    accepted directly for every other caller. Returns '_k_achieved' /
    '_N_used' the same way."""
    rng = rng or _fallback_rng()
    if target_k_key_pairs is not None:
        n_pulses = max(int(np.ceil(target_k_key_pairs / (2.0 / 9.0) * k_margin)), 500)
    V = sample_e91_channel(rng) if V is None else V

    a_choice, b_choice, r_a, r_b = run_e91(
        n_pulses,
        V=V,
        eve_mode=eve_mode,
        eve_intensity=eve_intensity,
        lam=lam,
        profile=profile,
        rng=rng,
        drift_amp=float((detector or {}).get("drift_amp", 0.0)),
        drift_cycles=float((detector or {}).get("drift_cycles", 1.0)),
        mean_burst=max(2000, int(n_pulses) // int(rng.choice([8, 16, 32, 64]))),
        noise_profile=(detector or {}).get("noise_profile", "stationary"),
        burst_multiplier=float((detector or {}).get("burst_multiplier", 3.0)),
    )
    _rec = dict(a=a_choice, b=b_choice, ra=r_a, rb=r_b)

    def _more_e91(n):
        a, b, ra, rb = run_e91(
            n,
            V=V,
            eve_mode=eve_mode,
            eve_intensity=eve_intensity,
            lam=lam,
            profile=profile,
            rng=rng,
            drift_amp=float((detector or {}).get("drift_amp", 0)),
            noise_profile=(detector or {}).get("noise_profile", "stationary"),
            burst_multiplier=float((detector or {}).get("burst_multiplier", 3.0)),
        )
        return dict(a=a, b=b, ra=ra, rb=rb)

    _rec = _extend_to_k(
        _rec,
        _more_e91,
        lambda r: np.isin(np.char.add(r["a"], r["b"]), [a + b for a, b in KEY_PAIRS]),
        target_k_key_pairs,
        n_pulses,
    )
    a_choice, b_choice, r_a, r_b = (_rec[k] for k in ("a", "b", "ra", "rb"))
    n_pulses = len(a_choice)
    _n_gen = int(n_pulses)
    if (
        truncate and target_k_key_pairs is not None
    ):  # review fix C3: exactly K key pairs
        _km = np.isin(
            np.char.add(a_choice, b_choice), [_a + _b for _a, _b in KEY_PAIRS]
        )
        _cs = np.cumsum(_km)
        if _cs[-1] >= target_k_key_pairs:
            _cut = int(np.searchsorted(_cs, target_k_key_pairs)) + 1
            a_choice, b_choice, r_a, r_b = (
                a_choice[:_cut],
                b_choice[:_cut],
                r_a[:_cut],
                r_b[:_cut],
            )
            n_pulses = _cut
    qber_trace, chsh_trace = windowed_traces(a_choice, b_choice, r_a, r_b, n_windows)
    # review E6: the SAME temporal statistics as BB84/BKM07 -- dispersion index and normalised jump energy of the key-error trace, computed from
    # per-window (errors, key pairs) counts so the binomial floor is known; the CHSH-trace jump energy is kept as a separate feature.
    _edges = np.linspace(0, n_pulses, n_windows + 1, dtype=int)
    _kp = np.isin(np.char.add(a_choice, b_choice), [_a + _b for _a, _b in KEY_PAIRS])
    _err = _kp & (r_a != -r_b)
    _wc = np.array([int(_kp[lo:hi].sum()) for lo, hi in zip(_edges[:-1], _edges[1:])])
    _we = np.array([int(_err[lo:hi].sum()) for lo, hi in zip(_edges[:-1], _edges[1:])])

    qber_key = window_qber(a_choice, b_choice, r_a, r_b, 0, n_pulses)
    chsh_s = window_chsh(a_choice, b_choice, r_a, r_b, 0, n_pulses)

    corrs = []
    for ac, bc in CHSH_PAIRS:
        m = (a_choice == ac) & (b_choice == bc)
        corrs.append(np.mean(r_a[m] * r_b[m]) if m.sum() else np.nan)
    per_pair_corr_spread = float(np.nanmax(corrs) - np.nanmin(corrs))

    sifted_rate = float(
        np.mean([((a_choice == ac) & (b_choice == bc)).sum() for ac, bc in KEY_PAIRS])
        / n_pulses
        * len(KEY_PAIRS)
    )

    # s_qber_residual -- physics-derived anisotropy detector.
    # Honest depolarisation obeys |S| = 2*sqrt(2)*(1-2Q); deviation from
    # that curve indicates an ANISOTROPIC disturbance (e.g. an entangling
    # probe), as opposed to ordinary white noise. Measured: ~0 for honest
    # channels AND for basis-averaged intercept-resend; +0.16 to +0.28 for
    # ancilla attacks.
    if not np.isnan(chsh_s) and not np.isnan(qber_key):
        s_pred = TSIRELSON_BOUND * (1 - 2 * qber_key)
        s_qber_residual = float(abs(chsh_s) - s_pred)
    else:
        s_qber_residual = np.nan

    features = {
        "qber_key": qber_key,
        "chsh_S": chsh_s,
        "s_deviation": abs(TSIRELSON_BOUND - abs(chsh_s))
        if not np.isnan(chsh_s)
        else np.nan,
        "s_qber_residual": s_qber_residual,
        "per_pair_corr_spread": per_pair_corr_spread,
        "jump_energy": float(
            _jump_energy_norm(_we, _wc)
        ),  # review E6: same definition as BB84/BKM07 (key-error trace)
        "chsh_jump_energy": float(_jump_energy_rel(chsh_trace)),
        "qber_dispersion": float(_dispersion_index(_we, _wc)),
        "spectral_entropy": float(spectral_entropy(qber_trace)),
        "autocorr_lag1": float(autocorr_lag1(qber_trace)),
        "h_qber_key": float(binary_entropy(qber_key))
        if not np.isnan(qber_key)
        else np.nan,
        "sifted_rate": sifted_rate,
        "attack_duty_cycle": eve_intensity if eve_mode != "none" else 0.0,
        "V": V,
        "r_secure_di": e91_di_secure_key_rate(qber_key, chsh_s)
        if not (np.isnan(qber_key) or np.isnan(chsh_s))
        else 0.0,
        "z_qber_key": _z_vs_baseline(
            qber_key,
            int(
                np.isin(
                    np.char.add(a_choice, b_choice), [_a + _b for _a, _b in KEY_PAIRS]
                ).sum()
            ),
            baseline,
        ),  # review fix B8
    }
    _key_mask = np.isin(
        np.char.add(a_choice, b_choice), [_a + _b for _a, _b in KEY_PAIRS]
    )
    features["_k_achieved"] = int(
        _key_mask.sum()
    )  # Draft 2, item 8/6: key pairs actually obtained
    features["_N_used"] = int(n_pulses)
    features["_N_generated"] = int(_n_gen)
    features["_N_trunc"] = int(n_pulses)
    features["_key_yield"] = features["_k_achieved"] / max(n_pulses, 1)
    features["_detection_yield"] = 1.0
    features["_loss_rate"] = 0.0
    if truncate and target_k_key_pairs is not None:
        assert features["_k_achieved"] == target_k_key_pairs
    if return_record:
        features["_record"] = (a_choice, b_choice, r_a, r_b)
    return features


E91_FEATURE_NAMES = [
    "qber_key",
    "chsh_S",
    "s_deviation",
    "s_qber_residual",
    "per_pair_corr_spread",
    "jump_energy",
    "spectral_entropy",
    "autocorr_lag1",
    "h_qber_key",
    "sifted_rate",
    "r_secure_di",
    "z_qber_key",
    "qber_dispersion",
    "chsh_jump_energy",
]


DISTANCE_RANGE_BKM = (0.0, 15.0)


DL_ATTACKS = {
    "bb84": ["clean", "intercept_resend", "pns", "blocking", "loss_manipulation"],
    "e91": [
        "clean",
        "intercept_resend",
        "ancilla",
        "extra_depolarisation",
    ],  # review B5/F4: it is white noise, not a loss attack
    # simulate_bkm07_pulse has one interception mechanism; symmetric vs
    # asymmetric is a DATASET-level distinction (eve_fwd == eve_ret or not),
    # the same convention Section 5's generate_datasets() uses.
    "bkm07": ["clean", "symmetric_attack", "asymmetric_attack"],
}


SEEDS = SeedBook(MASTER_SEED)


# ======================================================================
# 0. Configuration, reproducibility and file utilities
# ======================================================================
@dataclass
class Config:
    """All experiment settings; split randomness is independent of model seeds."""

    out: str = "out"
    data_seed: int = 20260913
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    protocols: tuple[str, ...] = ("bb84", "bkm07", "e91")
    classes: dict[str, list[str]] = field(
        default_factory=lambda: {
            p: ["honest", *names[1:]] for p, names in DL_ATTACKS.items()
        }
    )
    runs_per_class: int = 300
    stress_per_class: int = 30
    key_events: int = 2000
    windows: int = 50
    latent: int = 8
    hidden: int = 64
    ff_width: int = 128
    transformer_layers: int = 2
    dropout: float = 0.0
    pool_pairs: bool = False
    feature_set: str = "full"
    backbone: str = "lstm"
    epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    batch_size: int = 32
    lr: float = 1e-3
    pretrained_lr: float = 1e-4
    weight_decay: float = 1e-4
    folds: int = 5
    crossfit: bool = True
    alpha: float = 0.05
    beta: float = 0.02
    holdout_attack: str | None = None
    train_shuffle: bool = False
    device: str = "auto"
    threads: int = 2
    bootstrap: int = 1000
    smoke: bool = False
    label_budgets: tuple[int, ...] = (5, 20, 100, -1)
    shared_features: tuple[str, ...] = ("qber_key",)

    def known_classes(self, protocol: str) -> list[str]:
        return [name for name in self.classes[protocol] if name != self.holdout_attack]

    @property
    def torch_device(self) -> torch.device:
        choice = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )
        return torch.device(choice)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))


def log_event(cfg: Config, event: str, **details: Any) -> None:
    root = Path(cfg.out)
    root.mkdir(parents=True, exist_ok=True)
    entry = dict(event=event, config_hash=cfg.digest, time=time.time(), **details)
    with (root / "run_log.jsonl").open("a") as stream:
        stream.write(json.dumps(json_safe(entry)) + "\n")
    print(event, details, flush=True)


# ======================================================================
# 1. Observable window features, independent runs and group splits
# ======================================================================
FEATURE_GROUPS = {
    "error_rate": [
        "qber_key",
        "qber_z",
        "qber_x",
        "qber_zs",
        "qber_zsr",
        "ret_monitor",
        "qber_zc",
        "qber_xc",
    ],
    "detection": ["detection_rate", "accepted_fraction"],
    "gains": ["gain_mu", "gain_nu", "gain_vacuum"],
    "counts": ["key_count", "ctrl_count", "sift_count"],
    "chsh": [f"correlation_{a}_{b}" for a in ALICE_ANGLES for b in BOB_ANGLES]
    + ["chsh_S"],
}
FEATURES = [name for names in FEATURE_GROUPS.values() for name in names]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}
EVAL_ONLY = {
    "attack_family",
    "attack_strength",
    "intervention_round_start",
    "intervention_window_start",
    "simulator_noise_settings",
    "operating_regime_id",
    "regime_type",
}


def window_features(
    protocol: str, record: dict, windows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return numerator, denominator (T,F), and emitted-round edges (T+1,).

    Comparisons are treated as disclosed test bits in this simulation. An actual
    key-distribution implementation must disclose/sacrifice those bits. No Eve
    state, photon number, attack mask or simulator parameters are features.
    CHSH is a derived statistic: its support is the smallest setting count.
    """
    n = len(next(iter(record.values())))
    edges = np.linspace(0, n, windows + 1, dtype=int)
    num = np.zeros((windows, len(FEATURES)), dtype=np.float64)
    den = np.zeros_like(num)
    for t, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi == lo:
            continue
        r = {key: values[lo:hi] for key, values in record.items()}

        def put(name: str, numerator: float, denominator: float) -> None:
            j = FEATURE_INDEX[name]
            num[t, j], den[t, j] = numerator, denominator

        def error(name: str, mask: np.ndarray, mismatch: np.ndarray) -> None:
            put(name, np.count_nonzero(mask & mismatch), np.count_nonzero(mask))

        if protocol == "bb84":
            signal = r["k"] == 0
            key = r["sift"] & signal
            mismatch = r["bit_A"] != r["bit_B"]
            error("qber_key", key, mismatch)
            error("qber_z", key & (r["bas_A"] == 0), mismatch)
            error("qber_x", key & (r["bas_A"] == 1), mismatch)
            put("detection_rate", r["click"].sum(), hi - lo)
            put("accepted_fraction", key.sum(), signal.sum())
            put("key_count", key.sum(), 1)
            for j, name in enumerate(("gain_mu", "gain_nu", "gain_vacuum")):
                eligible = r["k"] == j
                put(name, (r["click"] & eligible).sum(), eligible.sum())
        elif protocol == "bkm07":
            survived = r["survived"]
            masks = {
                branch: survived & (r["round_type"] == branch)
                for branch in ("SIFT_KEY", "SIFT_MONITOR", "CTRL_Z", "CTRL_X")
            }
            key = masks["SIFT_KEY"]
            error("qber_key", key, r["bit_B"] != r["bit_A_final"])
            error("qber_zs", key, r["bit_A"] != r["bit_B"])
            error("qber_zsr", key, r["bit_B"] != r["bit_A_final"])
            error("ret_monitor", masks["SIFT_MONITOR"], r["bit_B"] != r["bit_A_final"])
            error("qber_zc", masks["CTRL_Z"], r["bit_A"] != r["bit_A_final"])
            error("qber_xc", masks["CTRL_X"], r["bit_A"] != r["bit_A_final"])
            put("detection_rate", survived.sum(), hi - lo)
            put("accepted_fraction", key.sum(), survived.sum())
            put("key_count", key.sum(), 1)
            put("ctrl_count", (masks["CTRL_Z"] | masks["CTRL_X"]).sum(), 1)
            put("sift_count", (key | masks["SIFT_MONITOR"]).sum(), 1)
        else:
            key = np.zeros(hi - lo, dtype=bool)
            for a, b in KEY_PAIRS:
                key |= (r["a"] == a) & (r["b"] == b)
            # Original singlet convention: Bob's outcome is flipped for key comparison.
            error("qber_key", key, r["ra"] != -r["rb"])
            put("key_count", key.sum(), 1)
            for a in ALICE_ANGLES:
                for b in BOB_ANGLES:
                    pair = (r["a"] == a) & (r["b"] == b)
                    put(
                        f"correlation_{a}_{b}",
                        (r["ra"][pair] * r["rb"][pair]).sum(),
                        pair.sum(),
                    )
            pair_ids = [FEATURE_INDEX[f"correlation_{a}_{b}"] for a, b in CHSH_PAIRS]
            support = min(den[t, pair_ids])
            if support > 0:
                value = sum(
                    sign * num[t, j] / den[t, j]
                    for sign, j in zip(CHSH_SIGNS, pair_ids)
                )
                put("chsh_S", value * support, support)
            # There is no optical loss/detector model for E91: those masks stay zero.
    return num, den, edges


def simulate_observations(
    cfg: Config, protocol: str, attack: str, seed: int, stress: bool = False
) -> tuple[dict, dict, dict]:
    """Call the original exact-K extractor; return observables and separate audit data."""
    rng = np.random.default_rng(seed)
    strength = (
        0.0 if attack == "honest" else float(np.exp(rng.uniform(np.log(0.01), 0)))
    )
    # The operating distribution is identical across labels. Stress shifts it for
    # every class, rather than marking only honest or only attacked sessions.
    distance = float(
        rng.uniform(
            *(DISTANCE_RANGE_BKM if protocol == "bkm07" else CHANNEL_DISTANCE_RANGE_KM)
        )
    )
    visibility = sample_e91_channel(rng)
    drift = (
        float(rng.uniform(0, HONEST_DRIFT_MAX))
        if rng.random() < HONEST_DRIFT_FRAC
        else 0.0
    )
    if stress:
        drift = float(rng.uniform(0.6, 1.0))
        visibility = float(rng.uniform(0.78, 0.85))
        if attack != "honest":
            strength = float(rng.uniform(0.01, 0.05))
    detector = dict(drift_amp=drift)
    mode = "none" if attack == "honest" else attack
    profile = "bursty" if mode == "intercept_resend" and rng.random() < 0.5 else "iid"
    common = dict(rng=rng, detector=detector, return_record=True)
    if protocol == "bb84":
        features = collect_bb84_features(
            distance_km=distance,
            eve_mode=mode,
            eve_intensity=strength,
            profile=profile,
            target_k_signal_bits=cfg.key_events,
            max_pulses=1_000_000 if cfg.smoke else 20_000_000,
            **common,
        )
        raw = features.pop("_record")
        record = {
            k: raw[k]
            for k in ("bit_A", "bit_B", "bas_A", "bas_B", "k", "click", "sift")
        }
    elif protocol == "bkm07":
        forward = strength
        backward = strength
        if mode == "asymmetric_attack":
            forward *= float(rng.uniform(0.1, 0.4))
            backward *= float(rng.uniform(0.6, 1.0))
        features = collect_bkm07_features(
            distance_km=distance,
            eve_mode="none"
            if mode == "none"
            else "symmetric"
            if mode == "symmetric_attack"
            else "asymmetric",
            eve_fwd=forward,
            eve_ret=backward,
            target_k_key_rounds=cfg.key_events,
            max_pulses=500_000 if cfg.smoke else 15_000_000,
            **common,
        )
        record = features.pop("_record")
    else:
        features = extract_e91_features(
            V=visibility,
            eve_mode=mode,
            eve_intensity=strength,
            profile=profile,
            target_k_key_pairs=cfg.key_events,
            **common,
        )
        record = dict(zip(("a", "b", "ra", "rb"), features.pop("_record")))
    assert features["_k_achieved"] == cfg.key_events
    metadata = dict(
        attack_family=attack,
        attack_strength=strength,
        intervention_round_start=0 if attack != "honest" else None,
        intervention_window_start=0 if attack != "honest" else None,
        onset_definition="Attack mechanism enabled at round zero; original simulator does not expose actual attack-hit times.",
        simulator_noise_settings=dict(
            distance_km=distance, visibility=visibility, **detector
        ),
        operating_regime_id="unseen_drift" if stress else "training_distribution",
        regime_type=("honest_drift" if attack == "honest" else "weak_attack_overlap")
        if stress
        else "train_regime",
    )
    return record, metadata, features


def split_groups(
    labels: np.ndarray, groups: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Stratified group allocation, 70/10/5/5/10; at least 20 groups/class."""
    result = {name: [] for name in ("train", "val", "cal_A", "cal_B", "test")}
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        ids = np.unique(groups[labels == label])
        if len(ids) < 20:
            raise ValueError(
                "At least 20 groups per class are needed for all five splits."
            )
        rng.shuffle(ids)
        n = len(ids)
        # Integer rounding is recorded in split tables, rather than silently dropping groups.
        sizes = [
            int(0.7 * n),
            int(0.1 * n),
            max(1, int(0.05 * n)),
            max(1, int(0.05 * n)),
        ]
        cuts = np.cumsum([0, *sizes, n - sum(sizes)])
        for (name, values), lo, hi in zip(result.items(), cuts[:-1], cuts[1:]):
            values.extend(ids[lo:hi].tolist())
    sets = [set(v) for v in result.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == len(np.unique(groups))
    return {name: np.flatnonzero(np.isin(groups, ids)) for name, ids in result.items()}


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    count_scale: np.ndarray
    fit_groups: list[str]

    @classmethod
    def fit(cls, num: np.ndarray, den: np.ndarray, groups: np.ndarray) -> Normalizer:
        mask = den > 0
        values = np.divide(num, den, out=np.zeros_like(num), where=mask)
        count = mask.sum(axis=(0, 1)).clip(1)
        mean = (values * mask).sum(axis=(0, 1)) / count
        variance = (((values - mean) ** 2) * mask).sum(axis=(0, 1)) / count
        return cls(
            mean,
            np.sqrt(variance).clip(1e-6),
            np.log1p(den).max(axis=(0, 1)).clip(1e-6),
            groups.tolist(),
        )

    def transform(self, num: np.ndarray, den: np.ndarray) -> dict[str, torch.Tensor]:
        mask = den > 0
        values = np.divide(num, den, out=np.zeros_like(num), where=mask)
        x = np.where(mask, (values - self.mean) / self.std, 0)
        counts = np.log1p(den) / self.count_scale
        return dict(
            x=torch.tensor(x, dtype=torch.float32),
            m=torch.tensor(mask, dtype=torch.float32),
            c=torch.tensor(counts, dtype=torch.float32),
            w=torch.tensor(mask.any(axis=-1)),
        )


class RunDataset(Dataset):
    """Only input tensors and labels. Evaluation metadata is structurally absent."""

    def __init__(self, tensors: dict[str, torch.Tensor], labels: np.ndarray):
        self.tensors = tensors
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{k: v[index] for k, v in self.tensors.items()},
            "y": self.labels[index],
        }


def generate(cfg: Config) -> None:
    root = Path(cfg.out) / "data"
    root.mkdir(parents=True, exist_ok=True)
    for protocol in cfg.protocols:
        path = root / protocol
        if (path / "arrays.npz").exists():
            raise FileExistsError(
                f"{path}: use a fresh --out to avoid mixing data and checkpoints."
            )
        path.mkdir(parents=True, exist_ok=True)
        numerators, denominators, edges_all, rows, evaluation = [], [], [], [], []
        classes = cfg.known_classes(protocol)
        for stress, count in (
            (False, cfg.runs_per_class),
            (True, cfg.stress_per_class),
        ):
            for attack in cfg.classes[protocol]:
                for i in range(count):
                    index = len(rows)
                    seed = SeedBook(cfg.data_seed).seed(
                        f"{protocol}/{stress}/{attack}", i
                    )
                    record, meta, original = simulate_observations(
                        cfg, protocol, attack, seed, stress
                    )
                    num, den, edges = window_features(protocol, record, cfg.windows)
                    run_id = f"{protocol}_{index:06d}"
                    record_path = path / "records" / f"{run_id}.npz"
                    record_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(record_path, **record)
                    numerators.append(num)
                    denominators.append(den)
                    edges_all.append(edges)
                    rows.append(
                        dict(
                            run_id=run_id,
                            protocol=protocol,
                            group_id=run_id,
                            scenario_group="stress" if stress else "main",
                            seed=seed,
                            label=classes.index(attack) if attack in classes else -1,
                        )
                    )
                    evaluation.append(dict(run_id=run_id, **meta))
                    # Exact original extracted values are retained for audit, not repeated
                    # across windows or admitted to the new model's feature schema.
                    save_json(path / "original_features" / f"{run_id}.json", original)
                log_event(
                    cfg,
                    "generated_class",
                    protocol=protocol,
                    attack=attack,
                    stress=stress,
                    runs=count,
                )
        runs = pd.DataFrame(rows)
        main = np.flatnonzero((runs.scenario_group == "main") & (runs.label >= 0))
        local = split_groups(
            runs.label.to_numpy()[main], runs.group_id.to_numpy()[main], cfg.data_seed
        )
        splits = {key: main[index] for key, index in local.items()}
        splits["test"] = np.concatenate(
            [
                splits["test"],
                np.flatnonzero((runs.scenario_group == "main") & (runs.label < 0)),
            ]
        )
        splits["stress"] = np.flatnonzero(runs.scenario_group == "stress")
        num, den = np.array(numerators), np.array(denominators)
        normalizer = Normalizer.fit(
            num[splits["train"]],
            den[splits["train"]],
            runs.group_id.to_numpy()[splits["train"]],
        )
        np.savez_compressed(
            path / "arrays.npz", num=num, den=den, edges=np.array(edges_all)
        )
        save_json(path / "normalizer.json", asdict(normalizer))
        save_json(path / "splits.json", splits)
        save_json(path / "eval_meta.json", evaluation)
        save_json(path / "config.json", dict(**asdict(cfg), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        runs.to_csv(path / "runs.csv", index=False)
        hashes = [hashlib.sha256(num[i].tobytes() + den[i].tobytes()).hexdigest() for i in range(len(runs))]
        save_json(path / "audit_manifest.json", dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arrays_sha256=hashlib.sha256((path / "arrays.npz").read_bytes()).hexdigest(),
            run_count=len(runs), unique_observation_hashes=len(set(hashes)),
            duplicate_observation_hashes=len(hashes)-len(set(hashes)),
            split_groups={name: runs.group_id.iloc[ids].tolist() for name, ids in splits.items()},
        ))
    save_json(root / "feature_schema.json", dict(names=FEATURES, groups=FEATURE_GROUPS))


@dataclass
class DataBundle:
    """Training-facing data; deliberately excludes evaluation-only metadata."""

    num: np.ndarray
    den: np.ndarray
    edges: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    splits: dict[str, np.ndarray]
    tensors: dict[str, torch.Tensor]
    normalizer: Normalizer


def load_data(cfg: Config, protocol: str) -> DataBundle:
    root = Path(cfg.out) / "data" / protocol
    saved = json.loads((root / "config.json").read_text())
    if saved.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Simulator source changed since generation; regenerate using a fresh --out")
    for name in ("data_seed", "key_events", "windows", "holdout_attack", "runs_per_class", "stress_per_class"):
        if saved[name] != getattr(cfg, name):
            raise ValueError(
                f"Data/config mismatch for {name}; use matching flags or a new --out."
            )
    raw = np.load(root / "arrays.npz", allow_pickle=False)
    manifest = json.loads((root / "audit_manifest.json").read_text())
    if manifest["arrays_sha256"] != hashlib.sha256((root / "arrays.npz").read_bytes()).hexdigest():
        raise ValueError("Generated arrays differ from audit manifest")
    runs = pd.read_csv(root / "runs.csv")
    splits = {
        k: np.array(v, dtype=int)
        for k, v in json.loads((root / "splits.json").read_text()).items()
    }
    split_ids = [set(runs.group_id.iloc[v]) for v in splits.values()]
    if sum(map(len, split_ids)) != len(set.union(*split_ids)):
        raise ValueError("A simulation run appears in multiple splits")
    if manifest["run_count"] != len(runs) or set.union(*split_ids) != set(runs.group_id):
        raise ValueError("Audit manifest or split coverage does not match run records")
    norm = json.loads((root / "normalizer.json").read_text())
    normalizer = Normalizer(
        np.array(norm["mean"]),
        np.array(norm["std"]),
        np.array(norm["count_scale"]),
        norm["fit_groups"],
    )
    assert set(normalizer.fit_groups) == set(runs.group_id.iloc[splits["train"]])
    return DataBundle(
        raw["num"],
        raw["den"],
        raw["edges"],
        runs.label.to_numpy(),
        runs.group_id.to_numpy(dtype=str),
        splits,
        normalizer.transform(raw["num"], raw["den"]),
        normalizer,
    )


def subset(
    tensors: dict[str, torch.Tensor], indices: np.ndarray
) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in tensors.items()}


def loader(
    data: DataBundle,
    indices: np.ndarray,
    cfg: Config,
    seed: int,
    tensors: dict[str, torch.Tensor] | None = None,
    shuffle: bool = False,
) -> DataLoader:
    ds = RunDataset(
        subset(data.tensors if tensors is None else tensors, indices),
        data.labels[indices],
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


# ======================================================================
# 2. Honest autoencoder, window classifier and temporal readouts
# ======================================================================
def reconstruction_loss(error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Run-balanced mean of valid-window reconstruction errors, shapes (B,T)."""
    per_run = (error * valid).sum(1) / valid.sum(1).clamp_min(1)
    usable = valid.any(1)
    return per_run[usable].mean() if usable.any() else per_run.sum() * 0


def anomaly_topk(
    error: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, list[list[int]]]:
    scores, indices = [], []
    for e, w in zip(error, valid):
        positions = torch.where(w)[0]
        if not len(positions):
            scores.append(e.sum() * 0)
            indices.append([])
            continue
        k = math.ceil(0.1 * len(positions))
        values, chosen = torch.topk(e[positions], k)
        scores.append(values.mean())
        indices.append(positions[chosen].tolist())
    return torch.stack(scores), indices


class HonestAE(nn.Module):
    """Protocol adapters/decoders around a shared window encoder; x,m,c: (B,T,F)."""

    def __init__(self, cfg: Config):
        super().__init__()
        width = len(FEATURES)
        self.adapters = nn.ModuleDict(
            {
                p: nn.Sequential(nn.Linear(3 * width, 32), nn.ReLU())
                for p in cfg.protocols
            }
        )
        self.encoder = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, cfg.latent)
        )
        self.decoders = nn.ModuleDict(
            {
                p: nn.Sequential(
                    nn.Linear(cfg.latent, 32), nn.ReLU(), nn.Linear(32, width)
                )
                for p in cfg.protocols
            }
        )

    def forward(
        self, batch: dict[str, torch.Tensor], protocol: str
    ) -> dict[str, torch.Tensor]:
        x, m, c = batch["x"], batch["m"], batch["c"]
        z = self.encoder(self.adapters[protocol](torch.cat((x, m, c), dim=-1)))
        reconstruction = self.decoders[protocol](z)
        residual = m * (x - reconstruction)
        error = residual.square().sum(-1) / m.sum(-1).clamp_min(1)
        return dict(z=z, r=residual, E=error)


class TemporalReadout(nn.Module):
    """Compacts holes before LSTM; Transformer masks missing windows explicitly."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.kind = cfg.backbone
        self.hidden = cfg.hidden
        self.pool_pairs = cfg.pool_pairs
        self.max_windows = cfg.windows
        if self.kind == "lstm":
            self.temporal = nn.LSTM(cfg.hidden, cfg.hidden, batch_first=True)
        elif self.kind == "transformer":
            self.position = nn.Parameter(torch.zeros(1, cfg.windows + 1, cfg.hidden))
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
            nn.init.normal_(self.position, std=0.02)
            nn.init.normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                cfg.hidden, 4, cfg.ff_width, dropout=cfg.dropout, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(
                layer, cfg.transformer_layers, enable_nested_tensor=False
            )
        elif self.kind == "flatten":
            self.temporal = nn.Linear(cfg.windows * cfg.hidden, cfg.hidden)
        elif self.kind == "mean":
            self.temporal = nn.Linear(cfg.hidden, cfg.hidden)
        else:
            raise ValueError(self.kind)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.pool_pairs:
            if h.shape[1] % 2:
                h = F.pad(h, (0, 0, 0, 1))
                valid = F.pad(valid, (0, 1))
            weights = valid.reshape(len(h), -1, 2)
            h = (h.reshape(len(h), -1, 2, self.hidden) * weights[..., None]).sum(
                2
            ) / weights.sum(2)[..., None].clamp_min(1)
            valid = weights.any(2)
        if self.kind == "lstm":
            # Packing requires a contiguous prefix, so compact arbitrary invalid holes.
            lengths = valid.sum(1)
            compacted = torch.zeros_like(h)
            for i in range(len(h)):
                compacted[i, : lengths[i]] = h[i, valid[i]]
            packed = nn.utils.rnn.pack_padded_sequence(
                compacted,
                lengths.clamp_min(1).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (last, _) = self.temporal(packed)
            return last[-1] * (lengths > 0)[:, None]
        if self.kind == "transformer":
            h = torch.cat((self.cls.expand(len(h), -1, -1), h), dim=1)
            h = h + self.position[:, : h.shape[1]]
            padding = torch.cat(
                (torch.zeros(len(h), 1, dtype=torch.bool, device=h.device), ~valid),
                dim=1,
            )
            return self.temporal(h, src_key_padding_mask=padding)[:, 0]
        if self.kind == "flatten":
            h = h * valid[..., None]
            h = F.pad(h, (0, 0, 0, self.max_windows - h.shape[1]))
            return self.temporal(h.flatten(1))
        average = (h * valid[..., None]).sum(1) / valid.sum(1)[:, None].clamp_min(1)
        return self.temporal(average)


class Detector(nn.Module):
    """Independent anomaly output and multiclass attack logits for each protocol."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ae = HonestAE(cfg)
        self.feature_set = cfg.feature_set
        f = len(FEATURES)
        width = 3 * f
        if cfg.feature_set != "none":
            width += cfg.latent
        if cfg.feature_set in ("latent_score", "full"):
            width += 1
        if cfg.feature_set == "full":
            width += f
        self.window_mlp = nn.Sequential(
            nn.Linear(width, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )
        self.readout = TemporalReadout(cfg)
        self.heads = nn.ModuleDict(
            {p: nn.Linear(cfg.hidden, len(cfg.known_classes(p))) for p in cfg.protocols}
        )
        self.register_buffer("input_features", torch.ones(f))

    def raw_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for key in ("x", "m", "c"):
            result[key] = result[key] * self.input_features
        result["w"] = result["w"] & result["m"].bool().any(-1)
        return result

    def forward(
        self, batch: dict[str, torch.Tensor], protocol: str, use_cache: bool = False
    ) -> dict[str, torch.Tensor]:
        batch = self.raw_inputs(batch)
        if self.feature_set == "none":
            x = batch["x"]
            aux = dict(
                z=x.new_zeros((*x.shape[:2], self.cfg.latent)),
                r=torch.zeros_like(x),
                E=x.new_zeros(x.shape[:2]),
            )
        elif use_cache:
            aux = {key: batch[key] for key in ("z", "r", "E")}
        else:
            aux = self.ae(batch, protocol)
        inputs = [batch["x"], batch["m"], batch["c"]]
        if self.feature_set != "none":
            inputs.append(aux["z"])
        if self.feature_set in ("latent_score", "full"):
            inputs.append(aux["E"][..., None])
        if self.feature_set == "full":
            inputs.append(aux["r"])
        embeddings = self.window_mlp(torch.cat(inputs, dim=-1))
        pooled = self.readout(embeddings, batch["w"])
        score, _ = anomaly_topk(aux["E"], batch["w"])
        return dict(
            logits=self.heads[protocol](pooled), score=score, valid=batch["w"], **aux
        )


# ======================================================================
# 3. Epoch rule, honest cross-fitting and supervised training
# ======================================================================
@dataclass
class EarlyStop:
    patience: int = 10
    delta: float = 1e-4
    best: float = float("inf")
    best_epoch: int = 0
    stale: int = 0
    state: dict | None = None

    def update(self, loss: float, epoch: int, model: nn.Module) -> bool:
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        if loss <= self.best - self.delta:
            self.best, self.best_epoch, self.stale = loss, epoch, 0
            self.state = copy.deepcopy(model.state_dict())
        else:
            self.stale += 1
        return self.stale >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.state is None:
            raise RuntimeError("No valid checkpoint")
        model.load_state_dict(self.state)


def shuffled_batch(
    batch: dict[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    result = {k: v.clone() for k, v in batch.items()}
    for i, w in enumerate(batch["w"]):
        positions = torch.where(w.cpu())[0]
        order = positions[torch.randperm(len(positions), generator=generator)]
        for key in ("x", "m", "c", "z", "r", "E"):
            if key in result:
                result[key][i, positions] = batch[key][i, order]
    return result


def fit_phase(
    model: Detector,
    bundles: dict[str, DataBundle],
    train_ids: dict[str, np.ndarray],
    cfg: Config,
    seed: int,
    phase: str,
    cache: dict[str, dict] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Balanced protocol batches; an epoch visits the largest training set once."""
    device = cfg.torch_device
    model.to(device)
    train_loaders = {
        p: loader(d, train_ids[p], cfg, seed, None if cache is None else cache[p], True)
        for p, d in bundles.items()
    }
    val_loaders = {}
    for p, d in bundles.items():
        ids = d.splits["val"]
        if phase == "honest":
            ids = ids[d.labels[ids] == 0]
        val_loaders[p] = loader(d, ids, cfg, seed)
    if any(
        len(v.dataset) == 0 for v in [*train_loaders.values(), *val_loaders.values()]
    ):
        raise ValueError("Training and validation sets must be nonempty")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    stopper = EarlyStop(cfg.patience, cfg.min_delta)
    history = []
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)

    def loss_for(batch: dict, protocol: str, training: bool) -> torch.Tensor:
        batch = {k: v.to(device) for k, v in batch.items()}
        if cfg.train_shuffle and training:
            batch = shuffled_batch(batch, generator)
        if phase == "honest":
            batch = model.raw_inputs(batch)
            aux = model.ae(batch, protocol)
            return reconstruction_loss(aux["E"], batch["w"])
        output = model(batch, protocol, use_cache=training and cache is not None)
        valid = output["valid"].any(1)
        if not valid.any():
            raise ValueError("Batch has no usable observations")
        return F.cross_entropy(output["logits"][valid], batch["y"][valid])

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        iterators = {p: iter(dl) for p, dl in train_loaders.items()}
        losses = []
        for _ in range(max(len(dl) for dl in train_loaders.values())):
            optimizer.zero_grad()
            protocol_losses = []
            for p, dl in train_loaders.items():
                try:
                    batch = next(iterators[p])
                except StopIteration:
                    iterators[p] = iter(dl)
                    batch = next(iterators[p])
                protocol_losses.append(loss_for(batch, p, True))
            loss = torch.stack(protocol_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = []
        with torch.no_grad():
            for p, dl in val_loaders.items():
                total = sum(
                    float(loss_for(batch, p, False)) * len(batch["y"]) for batch in dl
                )
                validation.append(total / len(dl.dataset))
        val_loss = float(np.mean(validation))
        history.append(
            dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss)
        )
        if stopper.update(val_loss, epoch, model):
            break
    stopper.restore(model)
    return dict(
        phase=phase,
        history=history,
        epochs_run=epoch,
        best_epoch=stopper.best_epoch,
        best_validation_loss=stopper.best,
        seconds=time.perf_counter() - start,
    )


def set_trainable(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def honest_features(
    model: Detector, data: DataBundle, protocol: str, cfg: Config
) -> dict[str, torch.Tensor]:
    model.eval().to(cfg.torch_device)
    collected = {key: [] for key in ("z", "r", "E")}
    with torch.no_grad():
        for batch in loader(data, np.arange(len(data.labels)), cfg, 0):
            batch = {k: v.to(cfg.torch_device) for k, v in batch.items()}
            output = model.ae(model.raw_inputs(batch), protocol)
            for key in collected:
                collected[key].append(output[key].cpu())
    return {key: torch.cat(values) for key, values in collected.items()}


def pretrain_crossfit(
    model: Detector,
    bundles: dict[str, DataBundle],
    cfg: Config,
    seed: int,
    target_only: bool = False,
) -> tuple[dict, dict, list]:
    """Fold AEs exclude each cached run's group. Full AE is inference-only for train."""
    initial = copy.deepcopy(model.state_dict())
    honest_ids = {
        p: d.splits["train"][d.labels[d.splits["train"]] == 0]
        for p, d in bundles.items()
    }

    def enable_ae(m: Detector) -> None:
        set_trainable(m, False)
        for p in bundles:
            set_trainable(m.ae.adapters[p], True)
            set_trainable(m.ae.decoders[p], True)
        if not target_only:
            set_trainable(m.ae.encoder, True)

    enable_ae(model)
    history = [fit_phase(model, bundles, honest_ids, cfg, seed, "honest")]
    caches = {
        p: {**d.tensors, **honest_features(model, d, p, cfg)}
        for p, d in bundles.items()
    }
    audit = {p: [] for p in bundles}
    if not cfg.crossfit:
        return caches, audit, history
    folds = {}
    for p, d in bundles.items():
        ids = d.splits["train"]
        rng = np.random.default_rng(seed)
        assignments = np.full(len(d.labels), -1, dtype=int)
        for label in np.unique(d.labels[ids]):
            group_ids = np.unique(d.groups[ids[d.labels[ids] == label]])
            rng.shuffle(group_ids)
            for i, g in enumerate(group_ids):
                assignments[d.groups == g] = i % cfg.folds
        if len(honest_ids[p]) < cfg.folds:
            raise ValueError("Too few honest groups for cross-fitting")
        folds[p] = assignments
    for fold in range(cfg.folds):
        seed_everything(seed)
        fold_model = Detector(cfg)
        fold_model.load_state_dict(initial)
        enable_ae(fold_model)
        fit_ids = {p: ids[folds[p][ids] != fold] for p, ids in honest_ids.items()}
        history.append(
            dict(
                fold=fold,
                **fit_phase(fold_model, bundles, fit_ids, cfg, seed, "honest"),
            )
        )
        for p, d in bundles.items():
            held = d.splits["train"][folds[p][d.splits["train"]] == fold]
            assert not set(d.groups[held]) & set(d.groups[fit_ids[p]])
            outputs = honest_features(fold_model, d, p, cfg)
            for key, value in outputs.items():
                caches[p][key][held] = value[held]
            audit[p].append(
                dict(
                    fold=fold,
                    trained_groups=d.groups[fit_ids[p]].tolist(),
                    cached_groups=d.groups[held].tolist(),
                )
            )
    return caches, audit, history


def model_path(cfg: Config, protocol: str, model_id: str, seed: int) -> Path:
    return Path(cfg.out) / "models" / protocol / model_id / f"seed{seed}"


def train_model(
    cfg: Config,
    protocol: str,
    model_id: str,
    seed: int,
    bundles: dict[str, DataBundle] | None = None,
    initial: Detector | None = None,
    transfer_mode: str | None = None,
    label_ids: dict[str, np.ndarray] | None = None,
) -> Detector:
    seed_everything(seed)
    bundles = bundles or {protocol: load_data(cfg, protocol)}
    model = initial or Detector(cfg)
    histories, audit, cache = [], {}, None
    if cfg.feature_set != "none":
        cache, audit, histories = pretrain_crossfit(
            model, bundles, cfg, seed, target_only=transfer_mode is not None
        )
    set_trainable(model, True)
    set_trainable(model.ae, False)
    optimizer = None
    if transfer_mode:
        set_trainable(model, False)
        new_parameters = []
        old_parameters = []
        for p in bundles:
            set_trainable(model.heads[p], True)
            new_parameters.extend(model.heads[p].parameters())
        if transfer_mode == "partial":
            layer = (
                model.readout.temporal.layers[-1]
                if cfg.backbone == "transformer"
                else model.readout.temporal
            )
            set_trainable(layer, True)
            old_parameters.extend(layer.parameters())
        optimizer = torch.optim.AdamW(
            [
                dict(params=new_parameters, lr=cfg.lr),
                dict(params=old_parameters, lr=cfg.pretrained_lr),
            ],
            weight_decay=cfg.weight_decay,
        )
    if model_id != "ae_only":
        train_ids = label_ids or {p: d.splits["train"] for p, d in bundles.items()}
        histories.append(
            fit_phase(
                model, bundles, train_ids, cfg, seed, "supervised", cache, optimizer
            )
        )
    root = model_path(cfg, protocol, model_id, seed)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state=model.cpu().state_dict(),
            config=asdict(cfg),
            seed=seed,
            model_id=model_id,
        ),
        root / "best.pt",
    )
    save_json(root / "history.json", histories)
    save_json(root / "crossfit_audit.json", audit)
    if cache is not None:
        torch.save(
            {
                p: {k: v for k, v in tensors.items() if k in ("z", "r", "E")}
                for p, tensors in cache.items()
            },
            root / "train_features.pt",
        )
    log_event(
        cfg,
        "trained",
        protocol=protocol,
        model_id=model_id,
        seed=seed,
        epochs=sum(h["epochs_run"] for h in histories),
        seconds=sum(h["seconds"] for h in histories),
    )
    return model


def load_model(cfg: Config, protocol: str, model_id: str, seed: int) -> Detector:
    checkpoint = torch.load(
        model_path(cfg, protocol, model_id, seed) / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    settings = Config(**checkpoint["config"])
    model = Detector(settings)
    model.load_state_dict(checkpoint["state"])
    return model.eval().to(cfg.torch_device)


def predict(
    model: Detector,
    data: DataBundle,
    ids: np.ndarray,
    protocol: str,
    cfg: Config,
    tensors: dict | None = None,
) -> dict[str, np.ndarray]:
    model.eval().to(cfg.torch_device)
    chunks = {key: [] for key in ("logits", "score", "valid", "E", "r")}
    with torch.no_grad():
        for batch in loader(data, ids, cfg, 0, tensors):
            batch = {k: v.to(cfg.torch_device) for k, v in batch.items()}
            output = model(batch, protocol)
            for key in chunks:
                chunks[key].append(output[key].cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


# ======================================================================
# 3.6. Calibration: cal_A temperature/anomaly, cal_B abstention
# ======================================================================
def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(
        torch.tensor(logits, dtype=torch.float64) / temperature, dim=-1
    ).numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.tensor(logits, dtype=torch.float64)
    truth = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe"
    )
    initial = float(F.cross_entropy(values, truth))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(values / log_temperature.clamp(-5, 5).exp(), truth)
        loss.backward()
        return loss

    optimizer.step(closure)
    result = float(log_temperature.detach().clamp(-5, 5).exp())
    # T=1 is an admissible fallback if numerical optimization fails to improve.
    return result if float(F.cross_entropy(values / result, truth)) <= initial else 1.0


def upper_threshold(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        raise ValueError("No honest calibration runs")
    return float(np.quantile(values, 1 - alpha, method="higher"))


def abstention_band(
    scores: np.ndarray, labels: np.ndarray, alpha: float, beta: float
) -> tuple[float, float]:
    honest, attacks = scores[labels == 0], scores[labels > 0]
    if not len(honest) or not len(attacks):
        raise ValueError("cal_B requires honest and attacked groups")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    highs = [t for t in candidates if np.mean(honest > t) <= alpha]
    lows = [t for t in candidates if np.mean(attacks < t) <= beta]
    high = min(highs) if highs else 1.0
    low = max(lows) if lows else 0.0
    # A crossed band is collapsed conservatively; this can only reduce NORMAL decisions.
    return float(min(low, high)), float(high)


def threshold_key(protocol: str, model_id: str, seed: int) -> str:
    return f"{protocol}/{model_id}/seed{seed}"


def lock_calibration(
    cfg: Config,
    protocol: str,
    model_id: str,
    seed: int,
    model: Detector | None = None,
    data: DataBundle | None = None,
) -> dict:
    data = data or load_data(cfg, protocol)
    model = model or load_model(cfg, protocol, model_id, seed)
    a, b = data.splits["cal_A"], data.splits["cal_B"]
    pa, pb = (
        predict(model, data, a, protocol, cfg),
        predict(model, data, b, protocol, cfg),
    )
    temperature = (
        1.0 if model_id == "ae_only" else fit_temperature(pa["logits"], data.labels[a])
    )
    high_anomaly = upper_threshold(pa["score"][data.labels[a] == 0], cfg.alpha)
    low, high = abstention_band(
        1 - probabilities(pb["logits"], temperature)[:, 0],
        data.labels[b],
        cfg.alpha,
        cfg.beta,
    )
    if model_id == "ae_only":
        low, high = 0.0, 1.0
    checkpoint = model_path(cfg, protocol, model_id, seed) / "best.pt"
    value = dict(
        temperature=temperature,
        anomaly_available=model.feature_set != "none",
        classifier_available=model_id != "ae_only",
        anomaly_threshold=high_anomaly,
        t_low=low,
        t_high=high,
        alpha=cfg.alpha,
        beta=cfg.beta,
        cal_A_groups=data.groups[a],
        cal_B_groups=data.groups[b],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        empirical_only=True,
        cal_A_honest_support=int(sum(data.labels[a] == 0)),
        cal_B_honest_support=int(sum(data.labels[b] == 0)),
        cal_B_attack_support=int(sum(data.labels[b] > 0)),
    )
    path = Path(cfg.out) / "locked_thresholds.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[threshold_key(protocol, model_id, seed)] = json_safe(value)
    save_json(path, existing)
    return value


def read_thresholds(cfg: Config, protocol: str, model_id: str, seed: int) -> dict:
    path = Path(cfg.out) / "locked_thresholds.json"
    if not path.exists():
        raise RuntimeError("Run calibrate before evaluating test or stress data")
    locked = json.loads(path.read_text())[threshold_key(protocol, model_id, seed)]
    checkpoint = model_path(cfg, protocol, model_id, seed) / "best.pt"
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != locked["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Checkpoint changed after calibration; recalibrate before evaluation"
        )
    return locked


# ======================================================================
# 4. Predeclared model matrix and run-summary baselines
# ======================================================================
MODEL_MATRIX = {
    "mlp_summary": ("none", "mean"),
    "noae_lstm": ("none", "lstm"),
    "noae_transformer": ("none", "transformer"),
    "ae_only": ("full", "mean"),
    "ae_mean": ("full", "mean"),
    "noae_mean": ("none", "mean"),
    "ae_lstm": ("full", "lstm"),
    "ae_transformer": ("full", "transformer"),
}


def run_summaries(data: DataBundle) -> np.ndarray:
    """Six predeclared summaries per feature; no fitting or metadata access."""
    mask = data.den > 0
    value = np.divide(data.num, data.den, out=np.zeros_like(data.num), where=mask)
    count = mask.sum(1).clip(1)
    mean = value.sum(1) / count
    std = np.sqrt(((value - mean[:, None]) ** 2 * mask).sum(1) / count)
    maximum = np.where(mask, value, -np.inf).max(1)
    maximum[~np.isfinite(maximum)] = 0
    log_count = np.log1p(data.den).mean(1)
    missing = 1 - mask.mean(1)
    rate = data.num.sum(1) / data.den.sum(1).clip(1)
    return np.concatenate(
        (mean, std, maximum, log_count, missing, rate), axis=1
    ).astype(np.float32)


class SummaryMLP(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6 * len(FEATURES), 64), nn.ReLU(), nn.Linear(64, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def calibration_metrics(labels: np.ndarray, prob: np.ndarray) -> dict:
    confidence = prob.max(1)
    correct = prob.argmax(1) == labels
    rows = []
    ece = 0.0
    for i in range(15):
        selected = (confidence >= i / 15) & (
            confidence < (i + 1) / 15 if i < 14 else confidence <= 1
        )
        if selected.any():
            accuracy, conf = (
                float(correct[selected].mean()),
                float(confidence[selected].mean()),
            )
            ece += selected.mean() * abs(accuracy - conf)
            rows.append(
                dict(
                    bin=i, count=int(selected.sum()), confidence=conf, accuracy=accuracy
                )
            )
    target = np.eye(prob.shape[1])[labels]
    return dict(
        nll=float(log_loss(labels, prob, labels=np.arange(prob.shape[1]))),
        brier=float(np.mean(np.sum((prob - target) ** 2, axis=1))),
        ece=float(ece),
        reliability=rows,
    )


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, positive: np.ndarray
) -> dict:
    truth = labels != 0
    both = len(np.unique(truth)) == 2
    return dict(
        auc=float(roc_auc_score(truth, scores)) if both else None,
        pr_auc=float(average_precision_score(truth, scores)) if truth.any() else None,
        honest_false_alert_rate=float(positive[~truth].mean())
        if (~truth).any()
        else None,
        attack_recall=float(positive[truth].mean()) if truth.any() else None,
        honest_support=int((~truth).sum()),
        attack_support=int(truth.sum()),
    )


def classification_metrics(labels: np.ndarray, prob: np.ndarray) -> dict:
    valid = labels >= 0
    y, p = labels[valid], prob[valid]
    prediction = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(prob.shape[1]), zero_division=0
    )
    return dict(
        accuracy=float(accuracy_score(y, prediction)),
        macro_f1=float(f1.mean()),
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion_matrix=confusion_matrix(
            y, prediction, labels=np.arange(prob.shape[1])
        ),
    )


def run_baselines(cfg: Config) -> None:
    """QBER, LR, validation-tuned RF/boosting, and early-stopped summary MLP."""
    import joblib

    for protocol in cfg.protocols:
        data = load_data(cfg, protocol)
        tr, va, ca, cb, te = [
            data.splits[s] for s in ("train", "val", "cal_A", "cal_B", "test")
        ]
        summaries = run_summaries(data)
        q = FEATURE_INDEX["qber_key"]
        qber = data.num[:, :, q].sum(1) / data.den[:, :, q].sum(1).clip(1)
        threshold = upper_threshold(qber[ca][data.labels[ca] == 0], cfg.alpha)
        locked_path = Path(cfg.out) / "locked_thresholds.json"
        locked = json.loads(locked_path.read_text()) if locked_path.exists() else {}
        locked[threshold_key(protocol, "qber", cfg.data_seed)] = dict(
            threshold=threshold,
            alpha=cfg.alpha,
            binary_only=True,
            cal_A_groups=data.groups[ca].tolist(),
            empirical_only=True,
        )
        save_json(locked_path, locked)
        baseline_root = Path(cfg.out) / "results" / protocol
        save_json(
            baseline_root / "qber.json",
            dict(
                config_hash=cfg.digest,
                threshold=threshold,
                metrics=binary_metrics(data.labels[te], qber[te], qber[te] > threshold),
                binary_only=True,
            ),
        )
        for seed in cfg.seeds:
            seed_everything(seed)
            choices = {
                "logistic": [
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=seed),
                    )
                ],
                "random_forest": [
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=d,
                        random_state=seed,
                        n_jobs=cfg.threads,
                    )
                    for d in (4, 8)
                ],
            }
            try:
                from xgboost import XGBClassifier

                choices["boosting"] = [
                    XGBClassifier(
                        n_estimators=n,
                        max_depth=d,
                        n_jobs=cfg.threads,
                        random_state=seed,
                    )
                    for n, d in ((50, 3), (100, 5))
                ]
            except ImportError:
                choices["boosting"] = [
                    HistGradientBoostingClassifier(
                        max_iter=n, max_depth=d, random_state=seed
                    )
                    for n, d in ((50, 3), (100, 5))
                ]
            for name, models in choices.items():
                start = time.perf_counter()
                best, loss = None, float("inf")
                for model in models:
                    model.fit(summaries[tr], data.labels[tr])
                    validation = log_loss(
                        data.labels[va],
                        model.predict_proba(summaries[va]),
                        labels=np.arange(len(cfg.known_classes(protocol))),
                    )
                    if validation < loss:
                        best, loss = model, validation
                root = model_path(cfg, protocol, name, seed)
                root.mkdir(parents=True, exist_ok=True)
                joblib.dump(best, root / "model.joblib")
                probs = best.predict_proba(summaries)
                save_baseline_result(
                    cfg,
                    protocol,
                    name,
                    seed,
                    data,
                    probs,
                    dict(seconds=time.perf_counter() - start, val_nll=loss),
                )
            start = time.perf_counter()
            scaler = StandardScaler().fit(summaries[tr])
            x = torch.tensor(scaler.transform(summaries), dtype=torch.float32)
            y = torch.tensor(data.labels, dtype=torch.long)
            model = SummaryMLP(len(cfg.known_classes(protocol)))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            stopper = EarlyStop(cfg.patience, cfg.min_delta)
            history = []
            rng = np.random.default_rng(seed)
            for epoch in range(1, cfg.epochs + 1):
                model.train()
                for batch_ids in np.array_split(
                    rng.permutation(tr), math.ceil(len(tr) / cfg.batch_size)
                ):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(x[batch_ids]), y[batch_ids])
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = float(F.cross_entropy(model(x[va]), y[va]))
                history.append(dict(epoch=epoch, val_loss=val_loss))
                if stopper.update(val_loss, epoch, model):
                    break
            stopper.restore(model)
            with torch.no_grad():
                probs = probabilities(model(x).numpy())
            root = model_path(cfg, protocol, "mlp_summary", seed)
            root.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), root / "best.pt")
            joblib.dump(scaler, root / "normalizer.joblib")
            save_json(root / "history.json", history)
            save_baseline_result(
                cfg,
                protocol,
                "mlp_summary",
                seed,
                data,
                probs,
                dict(
                    epochs_run=epoch,
                    best_epoch=stopper.best_epoch,
                    seconds=time.perf_counter() - start,
                ),
            )


def save_baseline_result(
    cfg: Config,
    protocol: str,
    name: str,
    seed: int,
    data: DataBundle,
    prob: np.ndarray,
    training: dict,
) -> None:
    a, b, te = (data.splits[k] for k in ("cal_A", "cal_B", "test"))
    stress = data.splits["stress"]
    logits = np.log(np.clip(prob, 1e-12, 1))
    temperature = fit_temperature(logits[a], data.labels[a])
    calibrated = probabilities(logits, temperature)
    low, high = abstention_band(
        1 - calibrated[b, 0], data.labels[b], cfg.alpha, cfg.beta
    )
    locked_path = Path(cfg.out) / "locked_thresholds.json"
    locked = json.loads(locked_path.read_text()) if locked_path.exists() else {}
    locked[threshold_key(protocol, name, seed)] = dict(
        temperature=temperature,
        t_low=low,
        t_high=high,
        cal_A_groups=data.groups[a].tolist(),
        cal_B_groups=data.groups[b].tolist(),
        empirical_only=True,
        anomaly_available=False,
    )
    save_json(locked_path, locked)
    root = Path(cfg.out) / "results" / protocol / name
    save_json(
        root / f"seed{seed}.json",
        dict(
            config_hash=cfg.digest,
            seed=seed,
            training=training,
            thresholds=dict(temperature=temperature, t_low=low, t_high=high),
            classification=classification_metrics(data.labels[te], calibrated[te]),
            binary=binary_metrics(
                data.labels[te], 1 - calibrated[te, 0], 1 - calibrated[te, 0] > high
            ),
            stress=dict(
                classification=classification_metrics(data.labels[stress], calibrated[stress]),
                binary=binary_metrics(data.labels[stress], 1 - calibrated[stress, 0],
                                      1 - calibrated[stress, 0] > high),
            ),
        ),
    )


# ======================================================================
# 6. Evaluation, delay, occlusion and engineering diagnosis
# ======================================================================
def evaluate(
    cfg: Config,
    protocol: str,
    model_id: str,
    seed: int,
    detailed: bool = True,
    storage_protocol: str | None = None,
) -> dict:
    storage = storage_protocol or protocol
    model = load_model(cfg, storage, model_id, seed)
    data = load_data(cfg, protocol)
    locked = (
        read_thresholds(
            cfg,
            storage,
            model_id if storage == protocol else f"{model_id}_{protocol}",
            seed,
        )
        if storage == protocol
        else read_joint_thresholds(cfg, protocol, model_id, seed)
    )
    results = {}
    for split in ("test", "stress"):
        ids = data.splits[split]
        output = predict(model, data, ids, protocol, cfg)
        prob = probabilities(output["logits"], locked["temperature"])
        attack = 1 - prob[:, 0]
        valid = output["valid"].any(1)
        known = (data.labels[ids] >= 0) & valid
        result = dict(
            insufficient_observations=int((~valid).sum()),
            classification=classification_metrics(data.labels[ids][valid], prob[valid]),
            binary=binary_metrics(
                data.labels[ids][valid], attack[valid], attack[valid] > locked["t_high"]
            ),
            anomaly=binary_metrics(
                data.labels[ids][valid],
                output["score"][valid],
                output["score"][valid] > locked["anomaly_threshold"],
            ),
            calibration_before=calibration_metrics(
                data.labels[ids][known], probabilities(output["logits"][known])
            ),
            calibration_after=calibration_metrics(data.labels[ids][known], prob[known]),
        )
        if not locked["anomaly_available"]:
            result["anomaly"] = {
                "available": False,
                "reason": "No autoencoder in this baseline",
            }
        if model_id == "ae_only":
            result = {
                k: v
                for k, v in result.items()
                if k in ("anomaly", "insufficient_observations")
            }
        meta = pd.DataFrame(
            json.loads(
                (Path(cfg.out) / "data" / protocol / "eval_meta.json").read_text()
            )
        ).iloc[ids]
        result["regimes"] = {}
        for regime in meta.regime_type.unique():
            selection = (meta.regime_type.to_numpy() == regime) & valid
            regime_scores = output["score"] if model_id == "ae_only" else attack
            regime_threshold = (
                locked["anomaly_threshold"]
                if model_id == "ae_only"
                else locked["t_high"]
            )
            result["regimes"][regime] = binary_metrics(
                data.labels[ids][selection],
                regime_scores[selection],
                regime_scores[selection] > regime_threshold,
            )
        unknown = (data.labels[ids] < 0) & valid
        result["unfamiliar_attack"] = dict(
            family=cfg.holdout_attack,
            support=int(unknown.sum()),
            prevalence=float(unknown.mean()),
            strength=meta.attack_strength.to_numpy()[unknown],
            anomaly_recall=float(
                np.mean(output["score"][unknown] > locked["anomaly_threshold"])
            )
            if unknown.any() and locked["anomaly_available"]
            else None,
        )
        if detailed and model_id != "ae_only":
            result["selective"] = selective_metrics(data.labels[ids], prob, locked)
        results[split] = result
        destination = Path(cfg.out) / "results" / protocol / model_id
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / f"seed{seed}_{split}_predictions.npz",
            groups=data.groups[ids],
            labels=data.labels[ids],
            prob=prob,
            anomaly=output["score"],
            valid=valid,
        )
        if split == "test" and detailed:
            results["delay"] = prefix_delays(
                model, data, ids, protocol, cfg, locked, meta
            )
            if model_id != "ae_only":
                results["occlusion"] = occlusion_report(
                    model, data, ids, protocol, cfg, locked
                )
    histories = json.loads(
        (model_path(cfg, storage, model_id, seed) / "history.json").read_text()
    )
    results.update(
        config_hash=cfg.digest,
        seed=seed,
        training=histories,
        epochs_run=sum(h["epochs_run"] for h in histories),
        wall_time=sum(h["seconds"] for h in histories),
    )
    save_json(
        Path(cfg.out) / "results" / protocol / model_id / f"seed{seed}.json", results
    )
    log_event(cfg, "evaluated", protocol=protocol, model_id=model_id, seed=seed)
    return results


def selective_metrics(labels: np.ndarray, prob: np.ndarray, locked: dict) -> list[dict]:
    score = 1 - prob[:, 0]
    center = (locked["t_low"] + locked["t_high"]) / 2
    bands = [
        (max(0, center - width), min(1, center + width))
        for width in np.linspace(0, 0.5, 11)
    ]
    bands.append((locked["t_low"], locked["t_high"]))
    rows = []
    for lo, hi in bands:
        decided = (score < lo) | (score > hi)
        prediction = np.where(score < lo, 0, prob[:, 1:].argmax(1) + 1)
        known = decided & (labels >= 0)
        attack = (labels != 0) & decided
        rows.append(
            dict(
                t_low=lo,
                t_high=hi,
                coverage=float(decided.mean()),
                accuracy=float((prediction[known] == labels[known]).mean())
                if known.any()
                else None,
                selective_false_negative_rate=float((score[attack] < lo).mean())
                if attack.any()
                else None,
                all_attack_normal_rate=float(np.mean(score[labels != 0] < lo))
                if (labels != 0).any()
                else None,
            )
        )
    return rows


def prefix_delays(
    model: Detector,
    data: DataBundle,
    ids: np.ndarray,
    protocol: str,
    cfg: Config,
    locked: dict,
    metadata: pd.DataFrame,
) -> dict:
    classifier_flags, anomaly_flags = [], []
    per_prefix = []
    honest = data.labels[ids] == 0
    for length in range(1, cfg.windows + 1):
        tensors = {k: v.clone() for k, v in data.tensors.items()}
        for key in ("x", "m", "c", "w"):
            tensors[key][:, length:] = 0
        output = predict(model, data, ids, protocol, cfg, tensors)
        usable = output["valid"].any(1)
        flags = (
            1 - probabilities(output["logits"], locked["temperature"])[:, 0]
            > locked["t_high"]
        ) & usable
        anomalies = (output["score"] > locked["anomaly_threshold"]) & usable
        classifier_flags.append(flags)
        anomaly_flags.append(anomalies)
        per_prefix.append(
            dict(
                prefix=length,
                honest_false_alert_rate=float(flags[honest].mean()),
                anomaly_honest_false_alert_rate=float(anomalies[honest].mean()),
                cumulative_false_alert_rate=float(
                    np.any(classifier_flags, axis=0)[honest].mean()
                ),
                cumulative_anomaly_false_alert_rate=float(
                    np.any(anomaly_flags, axis=0)[honest].mean()
                ),
            )
        )
    result = {
        "prefixes": per_prefix,
        "onset_definition": "mechanism enabled at round zero; not latent first Eve hit",
    }
    for name, flags in [("classifier", classifier_flags), ("anomaly", anomaly_flags)]:
        if not locked[f"{name}_available"]:
            result[name] = {"available": False}
            continue
        matrix = np.array(flags).T
        detected = matrix.any(1)
        first = np.argmax(matrix, axis=1) + 1
        attacked = ~honest
        onset = metadata.intervention_window_start.fillna(0).to_numpy()
        delays = first - onset
        result[name] = dict(
            median_delay=float(np.median(delays[attacked & detected]))
            if (attacked & detected).any()
            else None,
            never_detected=int((attacked & ~detected).sum()),
            detected_by_prefix={
                str(t): float(np.mean(detected[attacked] & (first[attacked] <= t)))
                for t in (10, 20, 30, 40, 50)
                if t <= cfg.windows
            },
        )
    return result


def occlude(
    tensors: dict[str, torch.Tensor],
    features: list[int] = (),
    time_range: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    result = {k: v.clone() for k, v in tensors.items()}
    for key in ("x", "m", "c"):
        if len(features):
            result[key][:, :, list(features)] = 0
        if time_range is not None:
            result[key][:, time_range[0] : time_range[1]] = 0
    result["w"] = result["w"] & result["m"].bool().any(-1)
    return result


def occlusion_report(
    model: Detector,
    data: DataBundle,
    ids: np.ndarray,
    protocol: str,
    cfg: Config,
    locked: dict,
) -> list[dict]:
    variants = {
        name: occlude(data.tensors, [FEATURE_INDEX[f] for f in names])
        for name, names in FEATURE_GROUPS.items()
    }
    q = FEATURE_INDEX["qber_key"]
    variants["QBER_removed"] = occlude(data.tensors, [q])
    variants["QBER_only"] = occlude(
        data.tensors, [j for j in range(len(FEATURES)) if j != q]
    )
    for lo in range(0, cfg.windows - 9, 5):
        variants[f"windows_{lo}_{lo + 10}"] = occlude(
            data.tensors, time_range=(lo, lo + 10)
        )
    variants["shuffled_order"] = shuffled_batch(
        data.tensors, torch.Generator().manual_seed(cfg.data_seed)
    )

    def score(tensors: dict) -> dict:
        output = predict(model, data, ids, protocol, cfg, tensors)
        prob = probabilities(output["logits"], locked["temperature"])
        known = data.labels[ids] >= 0
        return dict(
            macro_f1=classification_metrics(data.labels[ids], prob)["macro_f1"],
            auc=binary_metrics(
                data.labels[ids], 1 - prob[:, 0], 1 - prob[:, 0] > locked["t_high"]
            )["auc"],
            true_class_probability=float(prob[known, data.labels[ids][known]].mean()),
        )

    reference = score(data.tensors)
    rows = []
    for name, tensors in variants.items():
        metrics = score(tensors)
        rows.append(
            dict(
                variant=name,
                **{
                    f"drop_{k}": reference[k] - metrics[k]
                    if reference[k] is not None and metrics[k] is not None
                    else None
                    for k in reference
                },
            )
        )
    return rows


def diagnose(
    model: Detector,
    batch: dict[str, torch.Tensor],
    protocol: str,
    locked: dict,
    round_edges: np.ndarray,
    classes: list[str],
) -> dict:
    """One run: x,m,c (1,T,F), w (1,T); JSON-serializable diagnostic."""
    if not batch["w"].any():
        return dict(
            protocol=protocol, decision="insufficient_observations", n_valid_windows=0
        )
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model({k: v.to(device) for k, v in batch.items()}, protocol)
    if not output["valid"].any():
        return dict(
            protocol=protocol, decision="insufficient_observations", n_valid_windows=0
        )
    _, top = anomaly_topk(output["E"], output["valid"])
    if not locked["anomaly_available"]:
        top = [[]]
    prob = probabilities(output["logits"].cpu().numpy(), locked["temperature"])[0]
    attack = float(1 - prob[0])
    score = float(output["score"][0])
    decision = (
        "NORMAL"
        if attack < locked["t_low"]
        else "ATTACK:" + classes[int(np.argmax(prob[1:])) + 1]
        if attack > locked["t_high"]
        else "UNCERTAIN"
    )
    windows = []
    residuals = []
    for t in top[0]:
        windows.append(
            dict(
                window=t,
                round_range=[int(round_edges[t]), int(round_edges[t + 1])],
                E=float(output["E"][0, t]),
            )
        )
        r = output["r"][0, t].cpu().numpy()
        available = batch["m"][0, t].bool().cpu().numpy()
        ranking = np.flatnonzero(available)[np.argsort(-(r[available] ** 2))[:3]]
        residuals.extend(
            dict(window=t, feature=FEATURES[j], signed_residual=float(r[j]))
            for j in ranking
        )
    return dict(
        protocol=protocol,
        n_valid_windows=int(output["valid"].sum()),
        anomaly_score=score if locked["anomaly_available"] else None,
        anomaly_threshold=locked["anomaly_threshold"]
        if locked["anomaly_available"]
        else None,
        anomaly_flag=score > locked["anomaly_threshold"]
        if locked["anomaly_available"]
        else None,
        top_windows=windows,
        top_residuals=residuals,
        class_probabilities=dict(zip(classes, prob.tolist())),
        p_attack=attack,
        decision=decision,
        note="flag = statistical deviation under simulated conditions",
    )


# ======================================================================
# 5. Pairwise transfer and balanced joint training
# ======================================================================
def sample_label_budget(data: DataBundle, budget: int, seed: int) -> np.ndarray:
    tr = data.splits["train"]
    if budget < 0:
        return tr.copy()
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(data.labels[tr]):
        ids = tr[data.labels[tr] == label]
        if len(ids) < budget:
            raise ValueError(f"Label budget {budget} exceeds class support {len(ids)}")
        selected.extend(rng.choice(ids, budget, replace=False))
    return np.sort(selected)


def transfer_experiments(cfg: Config) -> None:
    rows = []
    for source in cfg.protocols:
        for target in cfg.protocols:
            if source == target:
                continue
            for shared in (False, True):
                variant = "shared" if shared else "union"
                for seed in cfg.seeds:
                    source_id = f"source_{cfg.backbone}_{variant}"
                    seed_everything(seed)
                    source_model = Detector(cfg)
                    if shared:
                        source_model.input_features.zero_()
                        source_model.input_features[
                            [FEATURE_INDEX[n] for n in cfg.shared_features]
                        ] = 1
                    source_model = train_model(
                        cfg, source, source_id, seed, initial=source_model
                    )
                    data = load_data(cfg, target)
                    for budget in cfg.label_budgets:
                        try:
                            chosen = sample_label_budget(data, budget, seed)
                        except ValueError:
                            log_event(
                                cfg,
                                "transfer_budget_unavailable",
                                target=target,
                                budget=budget,
                            )
                            continue
                        metrics = {}
                        for method in ("scratch", "head_only", "partial"):
                            seed_everything(seed)
                            model = Detector(cfg)
                            model.input_features.copy_(source_model.input_features)
                            if method != "scratch":
                                # New adapter, decoder and target head; only shared learned modules transfer.
                                model.ae.encoder.load_state_dict(
                                    source_model.ae.encoder.state_dict()
                                )
                                model.window_mlp.load_state_dict(
                                    source_model.window_mlp.state_dict()
                                )
                                model.readout.load_state_dict(
                                    source_model.readout.state_dict()
                                )
                            name = f"transfer_{source}_{variant}_{cfg.backbone}_{budget}_{method}"
                            model = train_model(
                                cfg,
                                target,
                                name,
                                seed,
                                initial=model,
                                transfer_mode=None if method == "scratch" else method,
                                label_ids={target: chosen},
                            )
                            lock_calibration(cfg, target, name, seed, model, data)
                            result = evaluate(cfg, target, name, seed, detailed=False)
                            metrics[method] = result["test"]["classification"][
                                "macro_f1"
                            ]
                            rows.append(
                                dict(
                                    source=source,
                                    target=target,
                                    variant=variant,
                                    backbone=cfg.backbone,
                                    budget=budget,
                                    method=method,
                                    seed=seed,
                                    macro_f1=metrics[method],
                                    transfer_gain=metrics[method] - metrics["scratch"],
                                    auc=result["test"]["binary"]["auc"],
                                    attack_recall=result["test"]["binary"][
                                        "attack_recall"
                                    ],
                                    validation_labels=len(data.splits["val"]),
                                    calibration_labels=len(data.splits["cal_A"])
                                    + len(data.splits["cal_B"]),
                                    sampled_groups=data.groups[chosen].tolist(),
                                )
                            )
                            save_json(Path(cfg.out) / "tables" / "transfer.json", rows)
    pd.DataFrame(rows).to_csv(Path(cfg.out) / "tables" / "transfer.csv", index=False)


def read_joint_thresholds(cfg: Config, protocol: str, model_id: str, seed: int) -> dict:
    # Joint checkpoints are saved once, but thresholds are protocol specific.
    path = Path(cfg.out) / "locked_thresholds.json"
    if not path.exists():
        raise RuntimeError("Joint calibration is required")
    locked = json.loads(path.read_text())[threshold_key(protocol, model_id, seed)]
    checkpoint = model_path(cfg, "joint", model_id, seed) / "best.pt"
    if (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        != locked["checkpoint_sha256"]
    ):
        raise RuntimeError("Stale joint thresholds")
    return locked


def joint_experiment(cfg: Config) -> None:
    bundles = {p: load_data(cfg, p) for p in cfg.protocols}
    model_id = "joint_" + cfg.backbone
    for seed in cfg.seeds:
        model = train_model(cfg, "joint", model_id, seed, bundles=bundles)
        for p, data in bundles.items():
            # Reuse the calibrated protocol routine with a local checkpoint link.
            destination = model_path(cfg, p, model_id, seed)
            destination.mkdir(parents=True, exist_ok=True)
            source = model_path(cfg, "joint", model_id, seed)
            for filename in ("best.pt", "history.json"):
                link = destination / filename
                if link.is_symlink():
                    link.unlink()
                if link.exists():
                    raise FileExistsError(link)
                link.symlink_to((source / filename).resolve())
            lock_calibration(cfg, p, model_id, seed, model, data)
            evaluate(cfg, p, model_id, seed, detailed=False)
        total = sum(p.numel() for p in model.parameters())
        inactive = sum(
            p.numel()
            for name, module in [
                ("adapters", model.ae.adapters),
                ("decoders", model.ae.decoders),
                ("heads", model.heads),
            ]
            for p in module.parameters()
        )
        active = {
            p: total
            - inactive
            + sum(
                t.numel()
                for module in (
                    model.ae.adapters[p],
                    model.ae.decoders[p],
                    model.heads[p],
                )
                for t in module.parameters()
            )
            for p in cfg.protocols
        }
        save_json(
            Path(cfg.out) / "tables" / f"joint_size_seed{seed}.json",
            dict(total_parameters=total, active_per_protocol=active),
        )


# ======================================================================
# 6.3 and 8. Latent sensitivity, compact variants, latency and paired reports
# ======================================================================
def latent_sweep(cfg: Config) -> None:
    rows = []
    for latent in (4, 8, 16):
        setting = replace(cfg, latent=latent)
        for protocol in cfg.protocols:
            for seed in cfg.seeds:
                name = f"latent_{latent}_{cfg.backbone}"
                model = train_model(setting, protocol, name, seed)
                lock_calibration(setting, protocol, name, seed, model)
                history = json.loads(
                    (
                        model_path(setting, protocol, name, seed) / "history.json"
                    ).read_text()
                )
                # Selection uses honest validation only. Stress remains a final audit,
                # resolving the brief's conflicting "select on drift"/"never tune on stress" instructions.
                rows.append(
                    dict(
                        protocol=protocol,
                        seed=seed,
                        latent=latent,
                        honest_val_reconstruction=history[0]["best_validation_loss"],
                    )
                )
    table = pd.DataFrame(rows)
    destination = Path(cfg.out) / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "latent_validation.csv", index=False)
    selected = {
        p: int(
            table[table.protocol == p]
            .groupby("latent")
            .honest_val_reconstruction.mean()
            .idxmin()
        )
        for p in cfg.protocols
    }
    save_json(destination / "latent_selected.json", selected)
    for p, latent in selected.items():
        for seed in cfg.seeds:
            evaluate(
                replace(cfg, latent=latent),
                p,
                f"latent_{latent}_{cfg.backbone}",
                seed,
                detailed=False,
            )


def engineering_metrics(
    model: Detector, data: DataBundle, protocol: str, ids: np.ndarray
) -> dict:
    stored = io.BytesIO()
    torch.save(model.state_dict(), stored)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    original_device = next(model.parameters()).device
    model.eval().cpu()
    batch = subset(data.tensors, ids[:1])
    with torch.no_grad():
        for _ in range(20):
            model(batch, protocol)
        timings = []
        for _ in range(200):
            start = time.perf_counter()
            model(batch, protocol)
            timings.append(time.perf_counter() - start)
    model.to(original_device)
    torch.set_num_threads(previous)
    return dict(
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        state_dict_bytes=stored.tell(),
        cpu_batch_one_median_ms=1000 * float(np.median(timings)),
        warmup=20,
        repeats=200,
    )


def compact_experiments(cfg: Config) -> None:
    variants = {
        "reference": cfg,
        "pool_pairs": replace(cfg, pool_pairs=True),
        "mean": replace(cfg, backbone="mean"),
        "flatten": replace(cfg, backbone="flatten"),
        "dropout_01": replace(cfg, dropout=0.1),
        "dropout_02": replace(cfg, dropout=0.2),
        "hidden_32": replace(cfg, hidden=32),
        "ffn_64": replace(cfg, ff_width=64),
        "one_block": replace(cfg, transformer_layers=1),
    }
    rows = []
    for p in cfg.protocols:
        data = load_data(cfg, p)
        for name, setting in variants.items():
            if cfg.backbone != "transformer" and name in ("ffn_64", "one_block"):
                continue
            for seed in cfg.seeds:
                model_id = f"compact_{name}_{cfg.backbone}"
                model = train_model(setting, p, model_id, seed)
                va = data.splits["val"]
                prediction = predict(model, data, va, p, setting)
                rows.append(
                    dict(
                        protocol=p,
                        variant=name,
                        seed=seed,
                        val_macro_f1=classification_metrics(
                            data.labels[va], probabilities(prediction["logits"])
                        )["macro_f1"],
                        **engineering_metrics(model, data, p, va),
                    )
                )
        # Choose each independent knob using validation averages, then combine.
        table = pd.DataFrame(rows)
        current = table[table.protocol == p].groupby("variant").val_macro_f1.mean()
        combined = cfg
        for alternatives in (
            ("reference", "pool_pairs"),
            ("reference", "mean", "flatten"),
            ("reference", "dropout_01", "dropout_02"),
            ("reference", "hidden_32"),
            ("reference", "ffn_64"),
            ("reference", "one_block"),
        ):
            valid = [n for n in alternatives if n in current]
            choice = max(valid, key=lambda n: current[n])
            changes = {
                k: v for k, v in asdict(variants[choice]).items() if v != asdict(cfg)[k]
            }
            combined = replace(combined, **changes)
        for seed in cfg.seeds:
            model_id = "compact_combined_" + cfg.backbone
            model = train_model(combined, p, model_id, seed)
            lock_calibration(combined, p, model_id, seed, model, data)
            result = evaluate(combined, p, model_id, seed, detailed=False)
            rows.append(
                dict(
                    protocol=p,
                    variant="combined",
                    seed=seed,
                    test_macro_f1=result["test"]["classification"]["macro_f1"],
                    test_attack_recall=result["test"]["binary"]["attack_recall"],
                    **engineering_metrics(model, data, p, data.splits["test"]),
                )
            )
        save_json(Path(cfg.out) / "tables" / "compact.json", rows)
    pd.DataFrame(rows).to_csv(Path(cfg.out) / "tables" / "compact.csv", index=False)


def paired_bootstrap(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    differences = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        known = y[ids] >= 0
        labels = np.arange(a.shape[1])
        differences.append(
            f1_score(
                y[ids][known],
                a[ids][known].argmax(1),
                labels=labels,
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[ids][known],
                b[ids][known].argmax(1),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    return dict(
        metric="macro_f1_difference",
        estimate=float(
            f1_score(
                y[y >= 0],
                a[y >= 0].argmax(1),
                labels=np.arange(a.shape[1]),
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[y >= 0],
                b[y >= 0].argmax(1),
                labels=np.arange(b.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        ci95=np.quantile(differences, [0.025, 0.975]),
        resamples=repetitions,
        unit="test_group",
    )


def collect_tables(cfg: Config) -> None:
    rows = []
    pairs = []
    for protocol in cfg.protocols:
        root = Path(cfg.out) / "results" / protocol
        for path in root.glob("*/seed*.json"):
            result = json.loads(path.read_text())
            test = result.get("test", result)
            if "classification" not in test:
                continue
            rows.append(
                dict(
                    protocol=protocol,
                    model=path.parent.name,
                    seed=result["seed"],
                    macro_f1=test["classification"]["macro_f1"],
                    accuracy=test["classification"]["accuracy"],
                    auc=test["binary"]["auc"],
                    recall=test["binary"]["attack_recall"],
                    honest_false_alert_rate=test["binary"]["honest_false_alert_rate"],
                )
            )
        for backbone in ("lstm", "transformer"):
            for seed in cfg.seeds:
                paths = [
                    root / name / f"seed{seed}_test_predictions.npz"
                    for name in ("ae_" + backbone, "noae_" + backbone)
                ]
                if not all(p.exists() for p in paths):
                    continue
                a, b = [np.load(p, allow_pickle=False) for p in paths]
                assert np.array_equal(a["groups"], b["groups"])
                pairs.append(
                    dict(
                        protocol=protocol,
                        backbone=backbone,
                        seed=seed,
                        **paired_bootstrap(
                            a["labels"],
                            a["prob"],
                            b["prob"],
                            a["groups"],
                            cfg.bootstrap,
                            cfg.data_seed,
                        ),
                    )
                )
    destination = Path(cfg.out) / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(destination / "main.csv", index=False)
        table.groupby(["protocol", "model"]).agg(
            {
                name: ["mean", "std", "count"]
                for name in (
                    "macro_f1",
                    "accuracy",
                    "auc",
                    "recall",
                    "honest_false_alert_rate",
                )
            }
        ).to_csv(destination / "main_summary.csv")
    save_json(destination / "paired_bootstrap.json", pairs)


# ======================================================================
# 9. Self-tests: fail loudly before generating or fitting an experiment
# ======================================================================
def selftest() -> None:
    seed_everything(7)
    torch.set_num_threads(2)
    passed = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    n = 103
    record = dict(
        bit_A=np.zeros(n, dtype=int),
        bit_B=np.zeros(n, dtype=int),
        bas_A=np.zeros(n, dtype=int),
        bas_B=np.zeros(n, dtype=int),
        k=np.zeros(n, dtype=int),
        click=np.ones(n, dtype=bool),
        sift=np.ones(n, dtype=bool),
    )
    num, den, edges = window_features("bb84", record, 1)
    q = FEATURE_INDEX["qber_key"]
    vac = FEATURE_INDEX["gain_vacuum"]
    check(
        "1: valid zero differs from no opportunities",
        num[0, q] == 0 and den[0, q] == 103 and den[0, vac] == 0,
    )
    _, _, edges = window_features("bb84", record, 50)
    check(
        "2: emitted windows cover all rounds once",
        np.array_equal(
            np.concatenate([np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]),
            np.arange(n),
        ),
    )
    numerator = np.tile(num, (40, 3, 1))
    denominator = np.tile(den, (40, 3, 1))
    numerator[20:, :, q] = 10000
    groups = np.array([str(i) for i in range(40)])
    labels = np.repeat([0, 1], 20)
    splits = split_groups(labels, groups, 1)
    tr = splits["train"]
    normalizer = Normalizer.fit(numerator[tr], denominator[tr], groups[tr])
    check(
        "3: normalization uses training only",
        np.isclose(
            normalizer.mean[q], (numerator[tr, :, q] / denominator[tr, :, q]).mean()
        )
        and set(normalizer.fit_groups) == set(groups[tr]),
    )
    sets = [set(groups[i]) for i in splits.values()] + [{"stress_0"}]
    check(
        "4: all split groups and stress are disjoint",
        sum(map(len, sets)) == len(set.union(*sets)),
    )
    cfg = Config(
        windows=8,
        epochs=1,
        folds=2,
        protocols=("bb84",),
        classes={"bb84": ["honest", "intercept_resend"]},
    )
    tensor = dict(
        x=torch.randn(2, 5, len(FEATURES)),
        m=torch.ones(2, 5, len(FEATURES)),
        c=torch.ones(2, 5, len(FEATURES)),
        w=torch.ones(2, 5, dtype=torch.bool),
    )
    for backbone in ("lstm", "transformer"):
        model = Detector(replace(cfg, backbone=backbone)).eval()
        padded = {
            k: F.pad(v, (0, 0, 0, 3)) if v.ndim == 3 else F.pad(v, (0, 3))
            for k, v in tensor.items()
        }
        with torch.no_grad():
            a = model(tensor, "bb84")
            b = model(padded, "bb84")
            full_prefix = model(tensor, "bb84")
            empty_occlusion = model(occlude(tensor), "bb84")
        check(
            "5: " + backbone + " padding invariance",
            torch.allclose(a["logits"], b["logits"], atol=1e-6),
        )
        check(
            "12: " + backbone + " full prefix unchanged",
            torch.equal(a["logits"], full_prefix["logits"]),
        )
        check(
            "13: " + backbone + " empty occlusion unchanged",
            torch.equal(a["logits"], empty_occlusion["logits"]),
        )
    readout = TemporalReadout(cfg).eval()
    h = torch.randn(1, 5, 64)
    mask = torch.tensor([[True, False, True, False, False]])
    direct, _ = readout.temporal(h[:, [0, 2]])
    check(
        "6: LSTM uses last valid state through holes",
        torch.allclose(readout(h, mask), direct[:, -1], atol=1e-6),
    )
    ae = HonestAE(cfg)
    masked = {k: v.clone() for k, v in tensor.items()}
    masked["m"][:, :, 0] = 0
    masked["x"][:, :, 0] = 0
    aux = ae(masked, "bb84")
    check("7: masked residual is exactly zero", bool((aux["r"][:, :, 0] == 0).all()))
    check(
        "7b: run loss ignores invalid windows",
        torch.equal(
            reconstruction_loss(
                torch.tensor([[2.0, 1000.0]]), torch.tensor([[True, False]])
            ),
            torch.tensor(2.0),
        ),
    )
    # Exercise real cross-fitting, including recorded training/cached group provenance.
    transformed = normalizer.transform(numerator, denominator)
    tiny = DataBundle(
        numerator,
        denominator,
        np.tile(np.arange(4), (40, 1)),
        labels,
        groups,
        splits,
        transformed,
        normalizer,
    )
    cross_cfg = replace(cfg, windows=3)
    cross_model = Detector(cross_cfg)
    caches, audit, _ = pretrain_crossfit(cross_model, {"bb84": tiny}, cross_cfg, 3)
    check(
        "8: real cross-fit groups never overlap",
        all(
            not set(row["trained_groups"]) & set(row["cached_groups"])
            for row in audit["bb84"]
        ),
    )
    check(
        "8b: every training group receives an excluded-fold cache",
        set(g for row in audit["bb84"] for g in row["cached_groups"])
        == set(groups[tr]),
    )
    linear = nn.Linear(1, 1)
    stop = EarlyStop()
    stopped = 0
    for epoch in range(1, 60):
        with torch.no_grad():
            linear.weight.fill_(epoch)
        loss = 100 - epoch if epoch <= 37 else 63.0
        if stop.update(loss, epoch, linear):
            stopped = epoch
            break
    stop.restore(linear)
    check(
        "9: best 37 stops at 47 and restores 37",
        stopped == 47 and float(linear.weight.detach()[0, 0]) == 37,
    )
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    truth = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, truth)
    check(
        "10: temperature does not increase calibration NLL",
        log_loss(truth, probabilities(logits, temperature))
        <= log_loss(truth, probabilities(logits)),
    )
    for count in (1, 9, 10, 11, 50):
        scores, positions = anomaly_topk(
            torch.arange(count, dtype=torch.float32)[None],
            torch.ones(1, count, dtype=torch.bool),
        )
        check(
            f"11: top-k for {count} windows",
            len(positions[0]) == math.ceil(0.1 * count),
        )
    check(
        "14: flattened and mean projection parameters",
        sum(p.numel() for p in nn.Linear(50 * 64, 64).parameters()) == 204864
        and sum(p.numel() for p in nn.Linear(64, 64).parameters()) == 4160,
    )
    check(
        "15: schema excludes evaluation metadata",
        not set(FEATURES) & EVAL_ONLY and "eval_meta" not in RunDataset.__dict__,
    )
    # Physical branch conventions compared directly with exact source extractors.
    for protocol in ("bb84", "bkm07", "e91"):
        original_cfg = replace(cfg, smoke=True, key_events=16, windows=5)
        record, _, original = simulate_observations(
            original_cfg, protocol, "honest", 17
        )
        num, den, _ = window_features(protocol, record, 5)
        qber = num[:, q].sum() / den[:, q].sum()
        expected = original["qber_total" if protocol == "bb84" else "qber_key"]
        check("source parity: " + protocol + " key error", np.isclose(qber, expected))
        if protocol == "e91":
            check(
                "E91 optics are unavailable",
                not den[:, FEATURE_INDEX["detection_rate"]].any(),
            )
            all_num, all_den, _ = window_features(protocol, record, 1)
            j = FEATURE_INDEX["chsh_S"]
            check(
                "E91 CHSH exact signs/order",
                np.isclose(all_num[0, j] / all_den[0, j], original["chsh_S"]),
            )
    # With perfect measurement/return and faulty re-preparation, Bob keeps
    # Alice's bit while Alice receives the flipped carrier on about half the rounds.
    b = simulate_bkm07_batch(
        10000, 0.0, "none", 0.0, 0.0,
        rng=np.random.default_rng(19), eta_bob=1.0, e_detector=0.0,
        p_meas=0.0, p_ret=0.0, p_prep=0.5,
    )
    key = b["round_type"] == "SIFT_KEY"
    check("BKM Bob retains measured bit", bool(np.all(b["bit_B"][key] == b["bit_A"][key])))
    key_error = np.mean(b["bit_B"][key] != b["bit_A_final"][key])
    check("BKM preparation faults affect shared key", 0.45 < key_error < 0.55)
    num, den, _ = window_features("bkm07", b, 50)
    check("BKM key QBER matches disclosed bits", np.isclose(num[:, q].sum() / den[:, q].sum(), key_error))
    print(f"SELFTEST: {len(passed)} checks passed", flush=True)


# ======================================================================
# 7 and 10. Command line, outputs and tiered execution
# ======================================================================
def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "selftest",
            "gen",
            "baselines",
            "train",
            "calibrate",
            "evaluate",
            "transfer",
            "joint",
            "compact",
            "sweep_latent",
            "diagnose",
            "all",
        ),
    )
    parser.add_argument("--out", default="out")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="40 runs/class, K=64, 3 epochs, seeds 0 and 1; simulation physics unchanged",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("bb84", "bkm07", "e91"),
        default=["bb84", "bkm07", "e91"],
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-seed", type=int, default=20260913)
    parser.add_argument("--runs-per-class", type=int)
    parser.add_argument("--key-events", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--backbone", choices=("lstm", "transformer", "mean", "flatten"), default="lstm"
    )
    parser.add_argument(
        "--feature-set",
        choices=("none", "latent", "latent_score", "full"),
        default="full",
    )
    parser.add_argument("--latent", type=int, default=8)
    parser.add_argument("--no_crossfit", action="store_true")
    parser.add_argument("--holdout_attack")
    parser.add_argument("--train_shuffle", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--basic-evaluation",
        action="store_true",
        help="Skip expensive prefix/occlusion passes",
    )
    args = parser.parse_args()
    cfg = Config(
        out=args.out,
        protocols=tuple(args.protocols),
        data_seed=args.data_seed,
        device=args.device,
        threads=args.threads,
        backbone=args.backbone,
        feature_set=args.feature_set,
        latent=args.latent,
        crossfit=not args.no_crossfit,
        holdout_attack=args.holdout_attack,
        train_shuffle=args.train_shuffle,
        smoke=args.smoke,
    )
    if args.smoke:
        cfg = replace(
            cfg,
            runs_per_class=40,
            stress_per_class=2,
            key_events=64,
            epochs=3,
            seeds=(0, 1),
            label_budgets=(5, -1),
        )
    if args.seeds is not None:
        cfg.seeds = tuple(args.seeds)
    if args.runs_per_class is not None:
        cfg.runs_per_class = args.runs_per_class
    if args.key_events is not None:
        cfg.key_events = args.key_events
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if cfg.epochs < 1 or cfg.epochs > 100:
        raise ValueError("Epoch cap must be between 1 and 100")
    if cfg.runs_per_class < 20:
        raise ValueError("Need at least 20 runs/class for the five disjoint splits")
    if cfg.key_events < 1:
        raise ValueError("key-events must be positive")
    if cfg.holdout_attack == "honest":
        raise ValueError("Cannot hold out the honest class")
    if cfg.holdout_attack and not any(
        cfg.holdout_attack in cfg.classes[p] for p in cfg.protocols
    ):
        raise ValueError("Unknown held-out attack for selected protocols")
    if args.model_id is None:
        args.model_id = ("noae_" if cfg.feature_set == "none" else "ae_") + cfg.backbone
    return cfg, args


def run_neural(cfg: Config, name: str, detailed: bool = True) -> None:
    for protocol in cfg.protocols:
        for seed in cfg.seeds:
            model = train_model(cfg, protocol, name, seed)
            lock_calibration(cfg, protocol, name, seed, model)
            evaluate(cfg, protocol, name, seed, detailed=detailed)


def main() -> None:
    cfg, args = parse_args()
    torch.set_num_threads(cfg.threads)
    selftest()
    if args.command == "selftest":
        return
    # selftest sets a small CPU thread count; restore the requested setting.
    torch.set_num_threads(cfg.threads)
    save_json(Path(cfg.out) / "invocation_config.json", asdict(cfg))
    command = args.command
    if command == "gen":
        generate(cfg)
    elif command == "baselines":
        run_baselines(cfg)
    elif command == "train":
        for protocol in cfg.protocols:
            for seed in cfg.seeds:
                train_model(cfg, protocol, args.model_id, seed)
    elif command == "calibrate":
        for protocol in cfg.protocols:
            for seed in cfg.seeds:
                lock_calibration(cfg, protocol, args.model_id, seed)
    elif command == "evaluate":
        for protocol in cfg.protocols:
            for seed in cfg.seeds:
                evaluate(
                    cfg,
                    protocol,
                    args.model_id,
                    seed,
                    detailed=not args.basic_evaluation,
                )
        collect_tables(cfg)
    elif command == "transfer":
        transfer_experiments(cfg)
    elif command == "joint":
        joint_experiment(cfg)
    elif command == "compact":
        compact_experiments(cfg)
    elif command == "sweep_latent":
        latent_sweep(cfg)
    elif command == "diagnose":
        protocol = cfg.protocols[0]
        seed = cfg.seeds[0]
        data = load_data(cfg, protocol)
        index = args.run_index
        if index not in data.splits["test"] and index not in data.splits["stress"]:
            raise ValueError(
                "For an unbiased diagnostic, select a test or stress run index from splits.json"
            )
        model = load_model(cfg, protocol, args.model_id, seed)
        locked = read_thresholds(cfg, protocol, args.model_id, seed)
        result = diagnose(
            model,
            subset(data.tensors, np.array([index])),
            protocol,
            locked,
            data.edges[index],
            cfg.known_classes(protocol),
        )
        save_json(Path(cfg.out) / "diagnosis.json", result)
        print(json.dumps(result, indent=2))
    elif command == "all":
        generate(cfg)
        run_baselines(cfg)
        for name, (feature_set, backbone) in MODEL_MATRIX.items():
            if name == "mlp_summary":
                continue
            run_neural(
                replace(cfg, feature_set=feature_set, backbone=backbone),
                name,
                not args.basic_evaluation,
            )
        for feature_set in ("latent", "latent_score"):
            run_neural(
                replace(cfg, feature_set=feature_set),
                f"ablation_{feature_set}_{cfg.backbone}",
                False,
            )
        latent_sweep(cfg)
        transfer_experiments(cfg)
        joint_experiment(cfg)
        compact_experiments(cfg)
        collect_tables(cfg)
    log_event(cfg, "command_complete", command=command)


if __name__ == "__main__":
    main()
