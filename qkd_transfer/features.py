"""Common means comparable measurement definition, not equal distribution."""
from dataclasses import dataclass
import numpy as np


def features(data, schema):
    names = data.names
    if schema in ('qber', 'common', 'common_support'):
        j = names.index('qber_key')
        support = data.den[:, :, j:j+1]
        mask = support > 0
        qber = np.divide(data.num[:, :, j:j+1], support,
                         out=np.zeros_like(support), where=mask)
        x, m, selected = qber, mask, ['key_disagreement_rate']
        if schema != 'qber':
            delta = np.zeros_like(qber)
            dm = np.zeros_like(mask)
            delta[:, 1:] = qber[:, 1:] - qber[:, :-1]
            dm[:, 1:] = mask[:, 1:] & mask[:, :-1]
            x = np.concatenate([qber, delta, abs(delta)], axis=-1)
            m = np.concatenate([mask, dm, dm], axis=-1)
            selected += ['adjacent_error_change', 'absolute_error_change']
        if schema == 'common_support':
            x = np.concatenate([x, np.log1p(support)], axis=-1)
            m = np.concatenate([m, np.ones_like(mask)], axis=-1)
            selected += ['log_disclosed_key_comparisons']
    else:
        excluded = {'qber_zsr'}  # Exact duplicate of qber_key in corrected BKM07.
        if schema == 'specific':
            excluded.add('qber_key')
        if schema not in ('full', 'specific'):
            raise ValueError('Unknown feature schema: ' + schema)
        ids = [i for i, n in enumerate(names) if n not in excluded]
        den = data.den[:, :, ids]
        m = den > 0
        x = np.divide(data.num[:, :, ids], den, out=np.zeros_like(den), where=m)
        selected = [names[i] for i in ids]
    return np.where(m, x, 0).astype('float32'), m.astype(bool), selected


@dataclass
class Preprocessor:
    mode: str
    mean: np.ndarray
    std: np.ndarray
    fit_groups: list
    names: list

    @classmethod
    def fit(cls, x, mask, groups, names, mode='standard'):
        if not len(x):
            raise ValueError('No preprocessing reference runs')
        count = mask.sum(axis=(0, 1)).clip(1)
        mean = np.where(mask, x, 0).sum(axis=(0, 1)) / count
        var = np.where(mask, (x-mean)**2, 0).sum(axis=(0, 1)) / count
        if mode == 'raw':
            mean, std = np.zeros_like(mean), np.ones_like(mean)
        else:
            std = np.sqrt(var).clip(1e-5)
        return cls(mode, mean, std, list(groups), list(names))

    def transform(self, x, mask):
        if x.shape[-1] != len(self.names):
            raise ValueError('Incompatible feature schema')
        return np.where(mask, (x-self.mean)/self.std, 0).astype('float32')


def summaries(x, mask):
    """Run summaries; no completed-run value is replicated into temporal windows."""
    n = mask.sum(1).clip(1)
    mean = np.where(mask, x, 0).sum(1)/n
    std = np.sqrt(np.where(mask, (x-mean[:, None])**2, 0).sum(1)/n)
    low = np.where(mask, x, np.inf).min(1)
    high = np.where(mask, x, -np.inf).max(1)
    low[~np.isfinite(low)] = 0
    high[~np.isfinite(high)] = 0
    return np.concatenate([mean, std, low, high, mask.mean(1)], 1).astype('float32')


def summary_names(names):
    return [f'{stat}:{n}' for stat in ('mean', 'std', 'min', 'max', 'availability') for n in names]
