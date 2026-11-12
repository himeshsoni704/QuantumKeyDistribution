"""Post-hoc diagnostics never feed model selection or threshold calibration."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from .features import features, summaries, summary_names
from .evaluation import save_json, metrics, temperature_score


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob('metrics.json')):
        record = json.loads(path.read_text())
        rows.append({k: record[k] for k in ('regime', 'target', 'schema', 'model', 'representation', 'normalization', 'seed', 'split', 'method', 'budget_per_original_class')} |
                    {'source': '+'.join(record['source']), 'path': str(path.relative_to(root)),
                     **{k: v for k, v in record['metrics'].items() if isinstance(v, (int, float))}})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'results.csv', index=False)
    keys = ['regime', 'source', 'target', 'schema', 'model', 'representation', 'normalization', 'split', 'method', 'budget_per_original_class']
    quantities = ['macro_f1', 'attack_recall', 'honest_false_alarm_rate', 'auroc', 'average_precision', 'brier', 'ece']
    frame.groupby(keys, dropna=False)[quantities].agg(['mean', 'std', 'count']).to_csv(root/'summary.csv')
    gaps, asymmetry = [], []
    for _, row in frame[(frame.regime == 'zero_shot') & (frame.split == 'test')].iterrows():
        same = (frame.seed == row.seed) & (frame.model == row.model) & (frame.schema == row.schema) & (frame.normalization == row.normalization) & (frame.representation == row.representation)
        within = frame[same & (frame.regime == 'within') & (frame.source == row.source)]
        reverse = frame[same & (frame.regime == 'zero_shot') & (frame.source == row.target) & (frame.target == row.source) & (frame.split == 'test')]
        if len(within):
            gaps.append(dict(source=row.source, target=row.target, seed=row.seed, model=row.model, schema=row.schema,
                             normalization=row.normalization, representation=row.representation,
                             **{k+'_gap': float(within.iloc[0][k]-row[k]) for k in quantities}))
        if len(reverse):
            asymmetry.append(dict(source=row.source, target=row.target, seed=row.seed, model=row.model, schema=row.schema,
                                  normalization=row.normalization, representation=row.representation,
                                  **{k+'_forward_minus_reverse': float(row[k]-reverse.iloc[0][k]) for k in quantities}))
    pd.DataFrame(gaps).to_csv(root/'transfer_gaps.csv', index=False)
    pd.DataFrame(asymmetry).to_csv(root/'directional_asymmetry.csv', index=False)
    paired = []
    for path in root.rglob('paired_transfer_gain.json'):
        result = json.loads(path.read_text())
        paired.append(dict(path=str(path.relative_to(root)), **result['point'],
                           negative_transfer=result['point']['macro_f1'] < 0,
                           significantly_negative=result['bootstrap']['macro_f1']['ci95'][1] is not None and result['bootstrap']['macro_f1']['ci95'][1] < 0))
    if paired:
        pd.DataFrame(paired).to_csv(root/'paired_gains.csv', index=False)
        save_json(root/'negative_transfer.json', dict(comparisons=len(paired),
                  point_negative_fraction=float(np.mean([p['negative_transfer'] for p in paired])),
                  ci_negative_fraction=float(np.mean([p['significantly_negative'] for p in paired])),
                  caveat='Descriptive frequency across the configured comparisons, not independent hypothesis tests.'))
    efficiency = []
    for path in root.rglob('budget.json'):
        budget = json.loads(path.read_text())
        result_path = path.parent/'test/metrics.json'
        if result_path.exists():
            r = json.loads(result_path.read_text())
            efficiency.append(dict(target=r['target'], model=r['model'], schema=r['schema'], representation=r['representation'],
                                   normalization=r['normalization'], seed=r['seed'], **budget, **r['metrics']))
    if efficiency:
        pd.DataFrame(efficiency).drop(columns=['reliability_bins', 'selected_groups']).to_csv(root/'label_efficiency.csv', index=False)
    # Static scientific plots are conveniences, never inputs to model selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (model, schema, norm, representation), part in frame[(frame.regime == 'zero_shot') & (frame.split == 'test')].groupby(['model', 'schema', 'normalization', 'representation']):
        matrix = part.pivot_table(index='source', columns='target', values='auroc', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix.values, vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel('Target'); ax.set_ylabel('Source')
        ax.set_title('Frozen-source AUROC')
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                if np.isfinite(matrix.iloc[i, j]):
                    ax.text(j, i, f'{matrix.iloc[i,j]:.2f}', ha='center', va='center')
        fig.colorbar(im, ax=ax)
        destination = root/'plots'
        destination.mkdir(exist_ok=True)
        fig.savefig(destination/f'{model}_{schema}_{norm}_{representation}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)


def diagnose(payload, bundles, out, args):
    out = Path(out)
    diagnostic_config = {'shap': args.shap, 'explain_rows': args.explain_rows}
    if (out/'complete.json').exists() and json.loads((out/'complete.json').read_text()) == diagnostic_config:
        return
    source_names = payload['source_protocols']
    if len(source_names) != 1:
        return
    source = source_names[0]
    if payload['schema'] in ('full', 'specific'):
        # Explain full features within their own protocol; do not force invalid transfer.
        bundles = {source: bundles[source]}
    model, norm = payload['model'], payload['normalizer']
    rng = np.random.default_rng(payload['seed'])
    embeddings, values, masks = {}, {}, {}
    for p, data in bundles.items():
        x, m, names = features(data, payload['schema'])
        x = norm.transform(x, m)
        values[p], masks[p] = x, m
        embeddings[p] = model.embeddings(x, m, args, latent=model.kind in ('ae_lstm', 'jepa_lstm'))
    d = bundles[source]
    train = d.splits['train']
    attack_probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    attack_probe.fit(embeddings[source][train], d.binary[train])
    probe_rows = []
    for p, data in bundles.items():
        ids = data.splits['test']
        score = attack_probe.predict_proba(embeddings[p][ids])[:, 1]
        probe_rows.append(dict(source=source, target=p, metrics=metrics(data.binary[ids], score, .5)))
    save_json(out/'attack_probes.json', probe_rows)
    # Balanced protocol AND binary-state sampling, using separate run splits.
    ps = sorted(bundles)
    balanced = {}
    for split in ('train', 'test'):
        n = min(int((bundles[p].binary[bundles[p].splits[split]] == y).sum()) for p in ps for y in (0, 1))
        xs, ys = [], []
        for label, p in enumerate(ps):
            data = bundles[p]
            for attack in (0, 1):
                ids = data.splits[split][data.binary[data.splits[split]] == attack]
                chosen = rng.choice(ids, n, replace=False)
                xs.append(embeddings[p][chosen]); ys.extend([label]*n)
        balanced[split] = (np.concatenate(xs), np.asarray(ys))
    if len(ps) > 1:
        protocol_probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        protocol_probe.fit(*balanced['train'])
        save_json(out/'protocol_probe.json', dict(balanced_accuracy=balanced_accuracy_score(balanced['test'][1], protocol_probe.predict(balanced['test'][0])),
                                                  chance=1/len(ps), note='Post-hoc all-protocol probe, balanced by binary state; not part of zero-shot detector training.'))
    else:
        save_json(out/'protocol_probe.json', {'available': False, 'reason': 'Full/specific schemas are explained only within their protocol.'})
    # Unpaired embeddings: compare class means standardized using SOURCE TRAIN only.
    scale = StandardScaler().fit(embeddings[source][train])
    rows = []
    for p, data in bundles.items():
        for label in (0, 1):
            a = train[d.binary[train] == label]
            b = data.splits['test'][data.binary[data.splits['test']] == label]
            if len(a) and len(b):
                distance = np.linalg.norm(scale.transform(embeddings[source][a]).mean(0)-scale.transform(embeddings[p][b]).mean(0))
                rows.append(dict(target=p, binary_class=label, standardized_centroid_distance=float(distance)))
    save_json(out/'class_conditional_distances.json', rows)
    # Source-fitted PCA is descriptive, not evidence by itself.
    if embeddings[source].shape[1] >= 2:
        pca = PCA(n_components=2).fit(embeddings[source][train])
        points = []
        for p, data in bundles.items():
            ids = data.splits['test']
            xy = pca.transform(embeddings[p][ids])
            points.extend(dict(protocol=p, group=g, attack=int(y), pc1=float(a), pc2=float(b)) for g, y, (a, b) in zip(data.groups[ids], data.binary[ids], xy))
        pd.DataFrame(points).to_csv(out/'pca.csv', index=False)
    def explanation_ids(data):
        ids = data.splits['test']
        chosen = []
        for label in (0, 1):
            candidates = ids[data.binary[ids] == label]
            chosen.extend(rng.choice(candidates, min(len(candidates), max(1, args.explain_rows//2)), replace=False))
        return np.asarray(chosen, dtype=int)

    occlusions = []
    for p, data in bundles.items():
        ids = explanation_ids(data)
        x, m = values[p][ids], masks[p][ids]
        score = temperature_score(model.score(x, m, args), payload['calibration']['temperature'])
        base = metrics(data.binary[ids], score, payload['calibration']['threshold'])
        for j, name in enumerate(norm.names):
            altered = x.copy()
            altered[:, :, j] = altered[rng.permutation(len(altered)), :, j]
            score = temperature_score(model.score(altered, m, args), payload['calibration']['temperature'])
            value = metrics(data.binary[ids], score, payload['calibration']['threshold'])
            occlusions.append(dict(target=p, feature=name, macro_f1_drop=base['macro_f1']-value['macro_f1'],
                                   note='Permutation diagnostic; can break feature dependencies, not causal effect.'))
        shuffled, shuffled_mask = x.copy(), m.copy()
        for i in range(len(x)):
            order = rng.permutation(x.shape[1])
            shuffled[i], shuffled_mask[i] = x[i, order], m[i, order]
        score = temperature_score(model.score(shuffled, shuffled_mask, args), payload['calibration']['temperature'])
        occlusions.append(dict(target=p, feature='window_order', macro_f1_drop=base['macro_f1']-metrics(data.binary[ids], score, payload['calibration']['threshold'])['macro_f1']))
    save_json(out/'permutation.json', occlusions)
    if args.shap and model.kind in ('rf', 'boosting'):
        try:
            import shap
            background_ids = rng.choice(train, min(len(train), args.explain_rows), replace=False)
            source_background = summaries(values[source][background_ids], masks[source][background_ids])
            explainer = shap.TreeExplainer(model.estimator, data=source_background)
            importance, labels = {}, summary_names(norm.names)
            for p, data in bundles.items():
                ids = explanation_ids(data)
                x = summaries(values[p][ids], masks[p][ids])
                explanation = explainer(x, check_additivity=False)
                v = explanation.values
                if v.ndim == 3:
                    v = v[:, :, 1]
                importance[p] = abs(v).mean(0)
                np.savez_compressed(out/f'shap_{p}.npz', values=v, features=x, labels=data.binary[ids], feature_names=labels)
                import matplotlib.pyplot as plt
                shap.summary_plot(v, x, feature_names=labels, show=False)
                plt.savefig(out/f'shap_{p}.png', bbox_inches='tight'); plt.close()
                top = int(np.argmax(importance[p]))
                shap.dependence_plot(top, v, x, feature_names=labels, interaction_index=None, show=False)
                plt.savefig(out/f'shap_dependence_{p}.png', bbox_inches='tight'); plt.close()
            save_json(out/'shap_importance.json', dict(feature_names=labels, mean_absolute_shap=importance,
                      source_target_rank_correlation={p: float(spearmanr(importance[source], importance[p]).statistic) for p in ps},
                      caveat='SHAP explains fitted scores, not physical causality.'))
        except (ImportError, ValueError, TypeError, AttributeError) as error:
            save_json(out/'shap_unavailable.json', {'error': str(error), 'action': 'Install a compatible SHAP version or use saved permutation diagnostics.'})
    save_json(out/'complete.json', diagnostic_config)
