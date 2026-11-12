"""Locked calibration and run-group uncertainty. A low score is not a security proof."""
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import beta
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, brier_score_loss, log_loss)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False))
    os.replace(temp, path)


def calibrate_temperature(y, probability):
    logits = logit(np.clip(probability, 1e-6, 1-1e-6))
    if len(np.unique(y)) < 2:
        return 1.
    result = minimize_scalar(lambda t: log_loss(y, expit(logits/np.exp(t)), labels=[0, 1]),
                             bounds=(-3, 3), method='bounded')
    return float(np.exp(result.x))


def temperature_score(score, temperature):
    return expit(logit(np.clip(score, 1e-6, 1-1e-6))/temperature)


def honest_threshold(scores, alpha):
    """Conformal order statistic, strict > decision; no target FAR guarantee."""
    if not 0 < alpha < 1 or not len(scores):
        raise ValueError('Need honest calibration runs and 0 < alpha < 1')
    scores = np.sort(np.asarray(scores))
    rank = math.ceil((len(scores)+1)*(1-alpha))
    # Scores are bounded by 1. A threshold of 1 explicitly produces no alerts.
    threshold = 1. if rank > len(scores) else float(scores[rank-1])
    return dict(threshold=threshold, honest_calibration_count=len(scores), rank=rank,
                target_false_alarm_rate=alpha, insufficient_resolution=rank > len(scores),
                guarantee_scope='Marginal source honest exchangeability only; no guarantee under target shift')


def binomial_ci(k, n):
    if n == 0:
        return [None, None]
    return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k))]


def metrics(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    prediction = score > threshold
    honest, attack = y == 0, y == 1
    far = float(prediction[honest].mean()) if honest.any() else None
    recall = float(prediction[attack].mean()) if attack.any() else None
    two_classes = honest.any() and attack.any()
    weights = np.where(attack, .5/max(1, attack.sum()), .5/max(1, honest.sum()))
    bins = []
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (score >= lo) & ((score < hi) if hi < 1 else (score <= hi))
        if selected.any():
            conf, frequency = float(score[selected].mean()), float(y[selected].mean())
            ece += selected.mean()*abs(conf-frequency)
            bins.append(dict(lower=float(lo), upper=float(hi), count=int(selected.sum()), mean_score=conf, attack_frequency=frequency))
    return dict(macro_f1=float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
                accuracy=float(accuracy_score(y, prediction)), attack_recall=recall,
                honest_false_alarm_rate=far,
                auroc=float(roc_auc_score(y, score)) if two_classes else None,
                average_precision=float(average_precision_score(y, score)) if two_classes else None,
                balanced_prior_average_precision=float(average_precision_score(y, score, sample_weight=weights)) if two_classes else None,
                brier=float(brier_score_loss(y, score)), ece=float(ece), reliability_bins=bins,
                attack_prevalence=float(y.mean()), runs=len(y),
                honest_false_alarm_ci95=binomial_ci(int(prediction[honest].sum()), int(honest.sum())),
                attack_recall_ci95=binomial_ci(int(prediction[attack].sum()), int(attack.sum())))


def bootstrap(y, score, groups, threshold, count=1000, seed=0, other=None, other_threshold=None):
    """Resample complete run groups; optional paired metric differences."""
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    names = ('macro_f1', 'attack_recall', 'honest_false_alarm_rate', 'auroc', 'average_precision', 'brier', 'ece')
    values = {k: [] for k in names}
    for _ in range(count):
        ids = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        a = metrics(y[ids], score[ids], threshold)
        b = metrics(y[ids], np.asarray(other)[ids], other_threshold) if other is not None else None
        for k in names:
            if a[k] is not None and (b is None or b[k] is not None):
                values[k].append(a[k] if b is None else a[k]-b[k])
    return {k: dict(ci95=np.quantile(v, [.025, .975]).tolist() if v else [None, None],
                    valid_resamples=len(v)) for k, v in values.items()}
