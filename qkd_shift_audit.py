"""Evaluate fixed checkpoints on one-factor shifts without refitting anything."""
import argparse
from dataclasses import fields, replace
from pathlib import Path
import json
import numpy as np
import qkd_pipeline as q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', nargs='+', default=['noae_mean', 'ae_lstm', 'jepa_lstm'])
    parser.add_argument('--transfer', action='store_true', help='Evaluate all saved target transfer checkpoints')
    parser.add_argument('--runs-per-class', type=int, default=100)
    args = parser.parse_args()
    saved = json.loads(next((Path(args.out)/'data').glob('*/config.json')).read_text())
    cfg = q.Config(**{k:v for k,v in saved.items() if k in {f.name for f in fields(q.Config)}})
    cfg.out = args.out
    rows = []
    for protocol in cfg.protocols:
        data = q.load_data(cfg, protocol)
        model_ids = sorted(p.name for p in (Path(args.out)/'models'/protocol).glob('transfer_*')) if args.transfer else args.models
        if not model_ids:
            raise ValueError(f'No matching checkpoints for {protocol}')
        for suite in ('drift', 'strength', 'bursty', 'visibility', 'distance'):
            if (suite == 'visibility' and protocol != 'e91') or (suite == 'distance' and protocol == 'e91'):
                continue
            nums, dens, labels, groups, metadata = [], [], [], [], []
            setting = replace(cfg, stress_suite=suite)
            for label, attack in enumerate(cfg.known_classes(protocol)):
                for i in range(args.runs_per_class):
                    seed = q.SeedBook(cfg.data_seed).seed(f'isolated/{protocol}/{suite}/{attack}', i)
                    record, meta, _ = q.simulate_observations(setting, protocol, attack, seed, stress=True)
                    num, den, edges = q.window_features(protocol, record, cfg.windows)
                    nums.append(num); dens.append(den); labels.append(label)
                    groups.append(f'{protocol}/{suite}/{attack}/{i}'); metadata.append(meta)
            nums, dens, labels = np.array(nums), np.array(dens), np.array(labels)
            shifted = replace(data, num=nums, den=dens, labels=labels, groups=np.array(groups),
                              tensors=data.normalizer.transform(nums, dens))
            ids = np.arange(len(labels))
            for model_id in model_ids:
                for seed in cfg.seeds:
                    model = q.load_model(cfg, protocol, model_id, seed)
                    locked = q.read_thresholds(cfg, protocol, model_id, seed)
                    output = q.predict(model, shifted, ids, protocol, cfg)
                    prob = q.probabilities(output['logits'], locked['temperature'])
                    rows.append(dict(protocol=protocol, suite=suite, model=model_id, seed=seed,
                        classification=q.classification_metrics(labels, prob),
                        binary=q.binary_metrics(labels, 1-prob[:,0], 1-prob[:,0] > locked['t_high'])))
            q.save_json(Path(args.out)/'isolated_shifts'/f'{protocol}_{suite}_metadata.json', metadata)
            q.save_json(Path(args.out)/'isolated_shifts'/('transfer_results.json' if args.transfer else 'results.json'), rows)
            print(protocol, suite, 'completed', flush=True)

if __name__ == '__main__':
    main()
