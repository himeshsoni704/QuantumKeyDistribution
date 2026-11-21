#!/usr/bin/env python3
"""Diagnose saved QKD predictions, without model fitting or changing an experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def interval(successes, n):
    if not n:
        return np.nan, np.nan
    return (0.0 if successes == 0 else beta.ppf(.025, successes, n-successes+1),
            1.0 if successes == n else beta.ppf(.975, successes+1, n-successes))


def qber_reference(data_root, protocol, expected_hash):
    """Join by saved run identity; never by incidental row order."""
    folder = data_root / protocol
    arrays = folder / 'arrays.npz'
    if checksum(arrays) != expected_hash:
        raise ValueError(f'{protocol}: predictions belong to a different dataset')
    names = read_json(data_root / 'feature_schema.json')['names']
    feature = names.index('qber_key')
    with np.load(arrays, allow_pickle=False) as saved:
        numerator = saved['num'][:, :, feature]
        denominator = saved['den'][:, :, feature]
    rows = pd.read_csv(folder / 'runs.csv')
    if len(rows) != len(numerator):
        raise ValueError('Run metadata and arrays have different lengths')
    valid = denominator > 0
    ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=valid)
    mean_window = ratios.sum(1) / valid.sum(1).clip(1)
    pooled = np.divide(numerator.sum(1), denominator.sum(1),
                       out=np.full(len(rows), np.nan), where=denominator.sum(1) > 0)
    mean_window[~valid.any(1)] = np.nan
    groups = protocol + ':' + rows.group_id.astype(str)
    result = pd.DataFrame({'group': groups, 'original_label': rows.label,
                           'mean_window_qber': mean_window, 'pooled_qber': pooled})
    if result.group.duplicated().any():
        raise ValueError('Expected one row per independent simulation group')
    return result.set_index('group')


def read_predictions(folder):
    with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
        frame = pd.DataFrame({'group': saved['groups'].astype(str),
                              'label': saved['labels'], 'original_label': saved['original_labels'],
                              'score': saved['score']})
        threshold = float(saved['threshold'])
    if frame.group.duplicated().any():
        raise ValueError('Predictions contain repeated run groups')
    if not np.isfinite(frame.score).all() or not np.isfinite(threshold):
        raise ValueError('Nonfinite scores or threshold')
    if not np.array_equal(frame.label, (frame.original_label != 0).astype(int)):
        raise ValueError('Binary and original labels disagree')
    frame['alert'] = frame.score > threshold
    return frame, threshold


def per_attack(frame, names):
    rows = []
    for label, part in frame.groupby('original_label'):
        label = int(label)
        n = len(part)
        alerts = int(part.alert.sum())
        lo, hi = interval(alerts, n)
        attack = names[label] if 0 <= label < len(names) else f'unknown_attack_{label}'
        rows.append(dict(attack=attack, original_label=label, runs=n, alerts=alerts,
                         metric='honest_false_alarm_rate' if label == 0 else 'attack_recall',
                         rate=alerts/n, ci95_low=lo, ci95_high=hi))
    return rows


def draw_direction(frame, threshold, context, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, name, color in ((0, 'Honest', '#16827A'), (1, 'Attacked', '#BB4B43')):
        part = frame[frame.label == label]
        if part.empty:
            continue
        axes[0].hist(part.score, bins=np.linspace(0, 1, 26), weights=np.ones(len(part))/len(part),
                     alpha=.45, label=f'{name}, n={len(part)}', color=color)
        valid = part.mean_window_qber.dropna()
        if len(valid):
            axes[1].hist(valid, bins=np.linspace(0, 1, 26), weights=np.ones(len(valid))/len(valid),
                         alpha=.45, label=name, color=color)
    axes[0].axvline(threshold, color='black', linestyle='--', label=f'Source threshold {threshold:.4g}')
    axes[0].set_xlabel('Saved calibrated attack score')
    axes[1].set_xlabel('Raw mean of valid-window key QBERs')
    for ax in axes:
        ax.set_ylabel('Fraction of class per bin')
        ax.legend(fontsize=8)
    fig.suptitle(context)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=170)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def paired_comparisons(study, out, selected_models):
    rows = []
    metric_keys = ['macro_f1', 'attack_recall', 'honest_false_alarm_rate', 'auroc']
    for gain_path in sorted(study.rglob('paired_transfer_gain.json')):
        folder = gain_path.parent
        meta = read_json(folder / 'test' / 'metrics.json')
        if meta['model'] not in selected_models:
            continue
        budget = read_json(folder / 'budget.json')
        # The scratch checkpoint is a sibling of source_* in the same seed folder.
        seed_folder = folder.parent.parent
        scratch_folder = seed_folder / f"scratch_{meta['target']}_{budget['budget_per_original_class']}" / 'test'
        baseline = read_json(scratch_folder / 'metrics.json')
        transfer_frame, _ = read_predictions(folder / 'test')
        scratch_frame, _ = read_predictions(scratch_folder)
        a = transfer_frame.sort_values('group')
        b = scratch_frame.sort_values('group')
        if not np.array_equal(a.group, b.group) or not np.array_equal(a.original_label, b.original_label):
            raise ValueError('Paired gain does not use identical target test runs')
        gain = read_json(gain_path)
        row = dict(model=meta['model'], representation=meta['representation'], schema=meta['schema'],
                   normalization=meta['normalization'], source='+'.join(meta['source']), target=meta['target'],
                   seed=meta['seed'], method=budget['method'], budget=budget['budget_per_original_class'])
        for key in metric_keys:
            row[f'transfer_{key}'] = meta['metrics'][key]
            row[f'scratch_{key}'] = baseline['metrics'][key]
            row[f'gain_{key}'] = gain['point'][key]
            row[f'gain_{key}_ci_low'], row[f'gain_{key}_ci_high'] = gain['bootstrap'][key]['ci95']
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'paired_gains_by_seed.csv', index=False)
    if not frame.empty:
        keys = ['model', 'representation', 'schema', 'normalization', 'source', 'target', 'method', 'budget']
        frame['positive_f1'] = frame.gain_macro_f1 > 0
        frame['positive_f1_interval'] = frame.gain_macro_f1_ci_low > 0
        summary = frame.groupby(keys).agg(
            seeds=('seed', 'nunique'), mean_f1_gain=('gain_macro_f1', 'mean'),
            std_f1_gain=('gain_macro_f1', 'std'), positive_seeds=('positive_f1', 'sum'),
            positive_interval_seeds=('positive_f1_interval', 'sum'),
            transfer_recall=('transfer_attack_recall', 'mean'), scratch_recall=('scratch_attack_recall', 'mean'),
            transfer_false_alarms=('transfer_honest_false_alarm_rate', 'mean'),
            scratch_false_alarms=('scratch_honest_false_alarm_rate', 'mean'))
        summary.to_csv(out / 'paired_gain_consistency.csv')
    return len(frame)


def representation_diagnostics(study, out, selected_models):
    rows, missing = [], []
    for source_folder in sorted(study.glob('*/*/*/*/seed_*/source_*')):
        model = source_folder.parents[2].name
        if model not in selected_models:
            continue
        diagnostic = source_folder / 'diagnostics'
        identity = dict(checkpoint=str(source_folder.relative_to(study)), model=model)
        for filename in ('attack_probes.json', 'protocol_probe.json', 'class_conditional_distances.json'):
            path = diagnostic / filename
            if not path.exists():
                missing.append(dict(**identity, missing_file=filename))
                continue
            content = read_json(path)
            records = content if isinstance(content, list) else [content]
            for record in records:
                metrics = record.get('metrics', {})
                row = {k: v for k, v in record.items() if k != 'metrics'}
                row.update({f'probe_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))})
                rows.append(dict(**identity, diagnostic=filename, **row))
    pd.DataFrame(rows).to_csv(out / 'representation_diagnostics.csv', index=False)
    pd.DataFrame(missing).to_csv(out / 'missing_diagnostics.csv', index=False)
    return len(rows), len(missing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Existing primary study directory')
    parser.add_argument('--data-root', required=True, help='Original generated output or its data directory')
    parser.add_argument('--out', required=True, help='New report directory')
    parser.add_argument('--models', nargs='+', default=['mlp', 'logistic', 'ae_lstm', 'jepa_lstm'])
    args = parser.parse_args(argv)
    study, data_root, out = (Path(p).resolve() for p in (args.study, args.data_root, args.out))
    if (data_root / 'data' / 'feature_schema.json').exists():
        data_root = data_root / 'data'
    for original in (study, data_root):
        if out == original or out in original.parents or original in out.parents:
            raise ValueError('Choose a report directory outside the study and dataset directories')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty report directory to preserve previous reports')
    manifest = read_json(study / 'study.json')
    references = {}
    for protocol, provenance in manifest['datasets'].items():
        print('Verify data and compute raw QBER:', protocol, flush=True)
        references[protocol] = qber_reference(data_root, protocol, provenance['arrays_sha256'])
    out.mkdir(parents=True, exist_ok=True)
    plots = out / 'plots'
    plots.mkdir(exist_ok=True)
    attacks, distributions, inventory = [], [], []
    for metric_path in sorted(study.rglob('metrics.json')):
        meta = read_json(metric_path)
        if meta.get('model') not in args.models or meta.get('split') != 'test':
            continue
        if meta.get('regime') not in ('within', 'zero_shot', 'lopo', 'joint_shared_binary'):
            continue
        folder = metric_path.parent
        if not (folder / 'predictions.npz').exists():
            raise FileNotFoundError(f'Missing saved predictions: {folder}')
        frame, threshold = read_predictions(folder)
        if threshold != meta['threshold']:
            raise ValueError('Prediction and metric thresholds differ')
        target = meta['target']
        raw = references[target]
        if not frame.group.isin(raw.index).all():
            raise ValueError('Unknown simulation group in predictions')
        joined = frame.join(raw[['mean_window_qber', 'pooled_qber', 'original_label']], on='group', rsuffix='_data')
        if not np.array_equal(joined.original_label, joined.original_label_data):
            raise ValueError('Original attack labels differ from dataset')
        context = dict(regime=meta['regime'], source='+'.join(meta['source']), target=target,
                       model=meta['model'], schema=meta['schema'], representation=meta['representation'],
                       normalization=meta['normalization'], seed=meta['seed'])
        classes = manifest['datasets'][target]['config']['classes'][target]
        attacks.extend(dict(**context, threshold=threshold, **row) for row in per_attack(joined, classes))
        for label, part in joined.groupby('label'):
            distributions.append(dict(**context, binary_class=int(label), runs=len(part), threshold=threshold,
                                      fraction_above_threshold=float(part.alert.mean()),
                                      score_median=float(part.score.median()), score_q05=float(part.score.quantile(.05)),
                                      score_q95=float(part.score.quantile(.95)),
                                      mean_window_qber=float(part.mean_window_qber.mean()),
                                      pooled_qber=float(part.pooled_qber.mean())))
        if meta['regime'] == 'zero_shot':
            source_metric = folder.parent / 'within' / 'metrics.json'
            if source_metric.exists() and read_json(source_metric)['threshold'] != threshold:
                raise ValueError('Zero-shot threshold differs from within-source threshold')
            title = f"{context['source']} -> {target} | {meta['model']} | seed {meta['seed']}"
            filename = '__'.join(str(context[k]) for k in ('source','target','model','schema','representation','normalization','seed'))
            draw_direction(joined, threshold, title, plots / filename)
        inventory.append(dict(**context, path=str(metric_path.relative_to(study))))
        print('Reviewed', str(metric_path.relative_to(study)), flush=True)
    attack_frame = pd.DataFrame(attacks)
    attack_frame.to_csv(out / 'per_attack_by_seed.csv', index=False)
    if not attack_frame.empty:
        keys = ['regime','source','target','model','schema','representation','normalization','attack','metric']
        attack_frame.groupby(keys).agg(mean_rate=('rate','mean'), std_across_seeds=('rate','std'),
                                       seeds=('seed','nunique'), runs_per_seed=('runs','min')).to_csv(out / 'per_attack_summary.csv')
    pd.DataFrame(distributions).to_csv(out / 'score_and_qber_summary.csv', index=False)
    pd.DataFrame(inventory).to_csv(out / 'evaluation_inventory.csv', index=False)
    pairs = paired_comparisons(study, out, args.models)
    diagnostics, missing = representation_diagnostics(study, out, args.models)
    provenance = dict(study_sha256=checksum(study/'study.json'), models=args.models,
                      datasets={p: d['arrays_sha256'] for p,d in manifest['datasets'].items()},
                      report_script_sha256=checksum(__file__), evaluations=len(inventory),
                      paired_comparisons=pairs, representation_records=diagnostics, missing_diagnostic_files=missing)
    (out/'report.json').write_text(json.dumps(provenance, indent=2))
    directions = {(r['source'],r['target']) for r in inventory if r['regime']=='zero_shot'}
    text = [ '# Saved-output evaluation', '',
             f'Reviewed {len(inventory)} test evaluations and {pairs} paired comparisons.',
             f'Observed {len(directions)} directed zero-shot protocol combinations.', '',
             'Plots compare class-normalized saved scores and raw mean-window QBER. The dashed line is the original source threshold.',
             'Pooled QBER is also exported. Missing comparisons are excluded rather than interpreted as zero error.',
             'Per-attack intervals are exact binomial intervals for independent runs at a fixed threshold, conditional on the fitted model.',
             'Across-seed rows reuse the held-out runs. They are not extra independent test observations.',
             'Positive-seed counts describe consistency. They are not a cross-seed significance test.',
             'MLP head_only and partial coincide in this architecture. Do not count them as independent evidence.', '',
             f'Representation records found: {diagnostics}. Missing diagnostic files: {missing}.',
             'No representations are recomputed here. Missing diagnostics must be generated using the saved-checkpoint analyze command.',
             'Probe results are post-hoc diagnostics. Protocol predictability and attack predictability can coexist.', '',
             '## Files',
             '- plots/: one score/QBER plot pair for each saved zero-shot evaluation, in PNG and PDF.',
             '- per_attack_summary.csv: attack-specific recall and honest false alarms across model seeds.',
             '- paired_gain_consistency.csv: gains together with absolute scratch/transfer recall and false alarms.',
             '- representation_diagnostics.csv and missing_diagnostics.csv: existing probe results and gaps.' ]
    (out/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()
