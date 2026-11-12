"""All target labels are consumed after source fit/calibration for strict transfer."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
from .data import label_budget, sha256
from .features import features, Preprocessor
from .models import BinaryModel, seed_all
from .evaluation import (save_json, calibrate_temperature, temperature_score,
                         honest_threshold, metrics, bootstrap)


def concatenate(parts):
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(len(parts[0])))


def inputs(data, ids, schema):
    x, m, names = features(data, schema)
    return x[ids], m[ids], data.binary[ids], data.groups[ids]


def collect(bundles, split, schema):
    return concatenate([inputs(d, d.splits[split], schema) for d in bundles])


def make_fit(bundles, kind, schema, normalization, representation, args, seed, selected=None, initial=None, mode='scratch', budget=None):
    # Caller passes only source bundles for strict zero-shot / LOPO.
    train = collect(bundles, 'train', schema) if selected is None else inputs(bundles[0], selected, schema)
    val = collect(bundles, 'val', schema)
    x, m, y, groups = train
    vx, vm, vy, vg = val
    names = features(bundles[0], schema)[2]
    if initial is None:
        ids = y == 0 if normalization == 'honest' else np.ones(len(y), bool)
        normalizer = Preprocessor.fit(x[ids], m[ids], groups[ids], names, normalization)
        seed_all(seed)
        model = BinaryModel(kind, x.shape[-1], representation)
    else:
        model, normalizer = copy.deepcopy(initial['model']), copy.deepcopy(initial['normalizer'])
    x, vx = normalizer.transform(x, m), normalizer.transform(vx, vm)
    started = time.monotonic()
    model.fit(x, m, y, vx, vm, vy, args, seed, mode=mode)
    calibration = {}
    for split in ('cal_A', 'cal_B'):
        a, mask, labels, g = collect(bundles, split, schema)
        calibration[split] = (model.score(normalizer.transform(a, mask), mask, args), labels, g)
    scores_a, labels_a, ga = calibration['cal_A']
    temperature = 1. if kind == 'qber' else calibrate_temperature(labels_a, scores_a)
    scores_b, labels_b, gb = calibration['cal_B']
    locked = honest_threshold(temperature_score(scores_b[labels_b == 0], temperature), args.alpha)
    locked.update(temperature=temperature, temperature_groups=ga.tolist(), threshold_groups=gb[labels_b == 0].tolist())
    if model.net is not None:
        model.net.cpu()
    return dict(model=model, normalizer=normalizer, calibration=locked, schema=schema,
                kind=kind, representation=representation, normalization=normalization,
                train_groups=groups.tolist(), validation_groups=vg.tolist(),
                label_usage=dict(supervised_train=len(y), validation=len(vy), temperature_calibration=len(labels_a),
                                 threshold_honest_reference=int((labels_b == 0).sum()),
                                 honest_pretraining=int((y == 0).sum()) if initial is None and kind in ('ae_lstm', 'jepa_lstm') else 0),
                source_protocols=[d.protocol for d in bundles], mode=mode, budget_per_original_class=budget,
                initialization_source=initial['source_protocols'] if initial else [],
                seconds=time.monotonic()-started, seed=seed)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    joblib.dump(payload, temp, compress=3)
    os.replace(temp, path)
    save_json(path.with_suffix('.sha256.json'), {'sha256': sha256(path)})


def cached_fit(directory, build):
    path = Path(directory)/'checkpoint.joblib'
    if path.exists():
        expected = json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
        if sha256(path) != expected:
            raise ValueError('Checkpoint checksum changed: ' + str(path))
        return joblib.load(path)
    print('Training', str(path.parent), flush=True)
    payload = build()
    print('Saved fit', str(path.parent), 'seconds=', round(payload['seconds'], 2), flush=True)
    save_checkpoint(path, payload)
    save_json(path.parent/'training.json', {k: v for k, v in payload.items() if k not in ('model', 'normalizer')})
    save_json(path.parent/'history.json', payload['model'].history)
    save_json(path.parent/'preprocessing.json', vars(payload['normalizer']))
    return payload


def evaluate(payload, data, directory, args, regime, split='test', reference=False):
    directory = Path(directory)
    if (directory/'metrics.json').exists() and (directory/'predictions.npz').exists():
        return json.loads((directory/'metrics.json').read_text())
    ids = data.splits[split]
    if not len(ids):
        return None
    x, m, names = features(data, payload['schema'])
    norm = payload['normalizer']
    if names != norm.names:
        raise ValueError('Source/target feature definitions differ')
    reference_groups = []
    if reference:
        # Explicit target-honest adaptation, disjoint from target test and never strict zero-shot.
        honest = data.splits['cal_A'][data.binary[data.splits['cal_A']] == 0]
        norm = Preprocessor.fit(x[honest], m[honest], data.groups[honest], names, 'honest')
        reference_groups = data.groups[honest].tolist()
    score = payload['model'].score(norm.transform(x[ids], m[ids]), m[ids], args)
    score = temperature_score(score, payload['calibration']['temperature'])
    threshold = payload['calibration']['threshold']
    y, groups = data.binary[ids], data.groups[ids]
    if regime in ('zero_shot', 'lopo'):
        used = set(payload['train_groups'] + payload['validation_groups'] + payload['calibration']['temperature_groups'] + payload['calibration']['threshold_groups'] + norm.fit_groups)
        if used.intersection(data.groups) or reference:
            raise ValueError('Strict transfer has target data in fitting/calibration')
    result = dict(regime=regime, source=payload.get('initialization_source') or payload['source_protocols'], target=data.protocol,
                  method=payload['mode'], budget_per_original_class=payload.get('budget_per_original_class'),
                  initialization_source=payload.get('initialization_source', []),
                  schema=payload['schema'], model=payload['kind'], representation=payload['representation'],
                  normalization=payload['normalization'], seed=payload['seed'], split=split,
                  threshold=threshold, calibration=payload['calibration'], label_usage=payload['label_usage'],
                  target_reference_groups=reference_groups, dataset=data.provenance,
                  metrics=metrics(y, score, threshold),
                  bootstrap=bootstrap(y, score, groups, threshold, args.bootstrap, payload['seed']))
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory/'predictions.tmp.npz'
    np.savez_compressed(temp, groups=groups, labels=y, original_labels=data.labels[ids], score=score, threshold=threshold)
    os.replace(temp, directory/'predictions.npz')
    save_json(directory/'metrics.json', result)
    return result


def study_fingerprint(args, bundles):
    config = {k: v for k, v in vars(args).items() if k not in ('command', 'resume', 'out', 'device', 'threads', 'shap', 'explain_rows', 'analysis_limit', 'shift_runs', 'shift_seed', 'shift_suites', 'allow_new_simulator')}
    files = sorted(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parent.parent/'run_transfer_study.py']
    code = {p.name: sha256(p) for p in files}
    return dict(configuration=config, code=code, datasets={p: d.provenance for p, d in bundles.items()})


def run_study(args, bundles):
    root = Path(args.out)
    manifest = study_fingerprint(args, bundles)
    path = root/'study.json'
    if path.exists():
        if not args.resume:
            raise FileExistsError('Study exists; pass --resume or choose a new --out')
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Study configuration/code/dataset changed; use a new output directory')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Output directory must be empty for a new study')
        save_json(path, manifest)
    protocols = list(bundles)
    for schema in args.schemas:
        for normalization in args.normalizations:
            for kind in args.models:
                if kind == 'qber' and (schema != 'qber' or normalization != 'raw'):
                    continue
                representations = args.representations if kind in ('ae_lstm', 'jepa_lstm') else ['raw']
                for representation in representations:
                    for seed in args.seeds:
                        tag = f'{schema}/{normalization}/{kind}/{representation}/seed_{seed}'
                        for source in protocols:
                            folder = root/tag/f'source_{source}'
                            print('Fit/resume', str(folder.relative_to(root)), flush=True)
                            payload = cached_fit(folder, lambda: make_fit([bundles[source]], kind, schema, normalization, representation, args, seed))
                            if 'within' in args.regimes:
                                evaluate(payload, bundles[source], folder/'within', args, 'within')
                            portable = schema not in ('full', 'specific')
                            for target in protocols:
                                if target == source:
                                    continue
                                target_data = bundles[target]
                                if portable and 'zero_shot' in args.regimes:
                                    evaluate(payload, target_data, folder/f'zero_shot_{target}', args, 'zero_shot')
                                    if 'stress' in target_data.splits:
                                        evaluate(payload, target_data, folder/f'stress_{target}', args, 'zero_shot', 'stress')
                                if portable and 'reference' in args.regimes and normalization == 'honest':
                                    evaluate(payload, target_data, folder/f'honest_reference_{target}', args, 'honest_reference', reference=True)
                                if 'few_shot' in args.regimes and kind in ('mlp', 'lstm', 'ae_lstm', 'jepa_lstm'):
                                    for budget in args.budgets:
                                        if budget == 0:
                                            if portable:
                                                evaluate(payload, target_data, folder/f'zero_shot_{target}', args, 'zero_shot')
                                            continue
                                        try:
                                            chosen = label_budget(target_data, budget, seed)
                                        except ValueError as error:
                                            save_json(folder/f'budget_{target}_{budget}_unavailable.json', {'reason': str(error)})
                                            continue
                                        for method in ('scratch', 'head_only', 'partial'):
                                            # Scratch is cached once per target, not once per source.
                                            destination = (root/tag/f'scratch_{target}_{budget}' if method == 'scratch'
                                                           else folder/f'few_{target}_{budget}_{method}')
                                            adapted = cached_fit(destination, lambda: make_fit([target_data], kind, schema, normalization, representation, args, seed, chosen,
                                                                                             None if method == 'scratch' else payload, method, budget))
                                            result = evaluate(adapted, target_data, destination/'test', args, 'few_shot')
                                            save_json(destination/'budget.json', dict(budget_per_original_class=budget, selected_groups=target_data.groups[chosen].tolist(), method=method,
                                                                                     initialization_source=source if method != 'scratch' else None))
                                            if method != 'scratch':
                                                scratch_path = root/tag/f'scratch_{target}_{budget}/test/predictions.npz'
                                                with np.load(scratch_path) as a, np.load(destination/'test/predictions.npz') as b:
                                                    if not np.array_equal(a['groups'], b['groups']):
                                                        raise ValueError('Paired comparison groups differ')
                                                    ma, mb = metrics(a['labels'], a['score'], float(a['threshold'])), metrics(b['labels'], b['score'], float(b['threshold']))
                                                    gains = {k: mb[k]-ma[k] for k in ('macro_f1', 'attack_recall', 'honest_false_alarm_rate', 'auroc')}
                                                    save_json(destination/'paired_transfer_gain.json', dict(point=gains, bootstrap=bootstrap(b['labels'], b['score'], b['groups'], float(b['threshold']), args.bootstrap, seed, a['score'], float(a['threshold']))))
                        if schema not in ('full', 'specific') and 'lopo' in args.regimes:
                            for target in protocols:
                                sources = [bundles[p] for p in protocols if p != target]
                                if len(sources) < 2:
                                    continue
                                folder = root/tag/f'lopo_{target}'
                                payload = cached_fit(folder, lambda: make_fit(sources, kind, schema, normalization, representation, args, seed))
                                evaluate(payload, bundles[target], folder/'test', args, 'lopo')
                        if 'joint' in args.regimes:
                            folder = root/tag/'joint_shared_binary'
                            payload = cached_fit(folder, lambda: make_fit(list(bundles.values()), kind, schema, normalization, representation, args, seed))
                            for target in protocols:
                                evaluate(payload, bundles[target], folder/target, args, 'joint_shared_binary')
    from .analysis import summarize
    summarize(root)
