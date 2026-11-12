"""Read existing sufficient-statistic arrays without changing or copying datasets."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class Runs:
    protocol: str
    num: np.ndarray
    den: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    splits: dict
    names: list
    provenance: dict

    @property
    def binary(self):
        # The simulator uses -1 for an unknown attack, NOT for honest operation.
        return (self.labels != 0).astype(np.int64)


def validate(data):
    n = len(data.labels)
    if data.num.shape != data.den.shape or data.num.ndim != 3:
        raise ValueError('Expected matching run x window x feature arrays')
    if data.num.shape[0] != n or data.num.shape[2] != len(data.names):
        raise ValueError('Feature schema/run count mismatch')
    if len(set(data.names)) != len(data.names) or len(data.groups) != n:
        raise ValueError('Duplicate feature names or wrong group length')
    if not np.isfinite(data.num).all() or not np.isfinite(data.den).all() or (data.den < 0).any():
        raise ValueError('Invalid sufficient statistics')
    owner, seen_rows = {}, set()
    for split, ids in data.splits.items():
        if len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n):
            raise ValueError('Duplicate or invalid split indices')
        for i in ids:
            g = str(data.groups[i])
            if g in owner and owner[g] != split:
                raise ValueError('Simulation group crosses splits')
            owner[g] = split
            seen_rows.add(int(i))
    if seen_rows != set(range(n)):
        raise ValueError('Splits must cover every run')
    if len(set(data.groups)) != n:
        raise ValueError('Expected one sequence per simulation run; aggregate repeated runs first')
    for split in ('train', 'val', 'cal_A', 'cal_B', 'test'):
        if split not in data.splits or not len(data.splits[split]):
            raise ValueError('Missing nonempty split: ' + split)
    # Exact duplicate observations across splits warrant investigation, not silent use.
    fingerprints = {}
    duplicates = 0
    for i in range(n):
        digest = hashlib.sha256(data.num[i].tobytes() + data.den[i].tobytes()).hexdigest()
        split = owner[str(data.groups[i])]
        if digest in fingerprints:
            duplicates += 1
        fingerprints.setdefault(digest, set()).add(split)
    data.provenance['exact_duplicate_rows'] = duplicates
    data.provenance['cross_split_duplicate_observations'] = sum(len(v) > 1 for v in fingerprints.values())
    # Identical low-count observations can occur by chance; IDs remain the independence unit.
    return data


def load_runs(root, protocol):
    root = Path(root)
    if (root / 'data' / 'feature_schema.json').exists():
        root = root / 'data'
    path = root / protocol
    names = json.loads((root / 'feature_schema.json').read_text())['names']
    config = json.loads((path / 'config.json').read_text())
    digest = sha256(path / 'arrays.npz')
    manifest_path = path / 'audit_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest['arrays_sha256'] != digest:
        raise ValueError('Dataset checksum differs from audit manifest: ' + protocol)
    with np.load(path / 'arrays.npz', allow_pickle=False) as raw:
        num, den = raw['num'].copy(), raw['den'].copy()
    rows = pd.read_csv(path / 'runs.csv')
    splits = {k: np.asarray(v, dtype=int) for k, v in json.loads((path / 'splits.json').read_text()).items()}
    if manifest is not None and manifest['run_count'] != len(rows):
        raise ValueError('Manifest run count mismatch')
    provenance = dict(arrays_sha256=digest, config=config, manifest_present=manifest is not None,
                      metadata_sha256={p: sha256(path / p) for p in ('config.json', 'runs.csv', 'splits.json')},
                      schema_sha256=sha256(root / 'feature_schema.json'),
                      simulator_source_sha256=config.get('source_sha256'),
                      semantics='Historical observations retain their generating simulator version; no physics correction is inferred.')
    return validate(Runs(protocol, num, den, rows.label.to_numpy(),
                         np.array([protocol + ':' + str(g) for g in rows.group_id]), splits, names, provenance))


def label_budget(data, budget, seed):
    if budget == 0:
        raise ValueError('Zero labels must use strict zero-shot, not adaptation')
    rng = np.random.default_rng(seed)
    train = data.splits['train']
    chosen = []
    for label in sorted(set(data.labels[train])):
        if label < 0:
            raise ValueError('Unknown attack in training split')
        ids = train[data.labels[train] == label].copy()
        rng.shuffle(ids)
        if budget > len(ids):
            raise ValueError(f'Budget {budget} exceeds class {label} training count {len(ids)}')
        chosen.extend(ids if budget < 0 else ids[:budget])
    return np.asarray(sorted(chosen), dtype=int)
