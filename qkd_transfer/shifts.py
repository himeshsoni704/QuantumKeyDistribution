"""Optional compact evaluation-only simulations; never regenerate the main dataset."""
from dataclasses import fields, replace
from pathlib import Path
import json
import joblib
import numpy as np
from .data import Runs, sha256
from .evaluation import save_json
from .experiments import evaluate


def run_shifts(args, bundles):
    import qkd_pipeline as q
    root = Path(args.out)
    current_source = sha256(q.__file__)
    for target, data in bundles.items():
        saved = data.provenance['config']
        if data.provenance['simulator_source_sha256'] != current_source and not args.allow_new_simulator:
            raise ValueError('Shift simulator differs from historical data. Use --allow-new-simulator to explicitly evaluate a cross-version shift; this is not a pure noise ablation.')
        cfg = q.Config(**{k: v for k, v in saved.items() if k in {f.name for f in fields(q.Config)}})
        for suite in args.shift_suites:
            if suite == 'visibility' and target != 'e91' or suite == 'distance' and target == 'e91':
                continue
            signature = dict(protocol=target, suite=suite, runs_per_class=args.shift_runs, seed=args.shift_seed,
                             simulator=current_source, source_config=saved)
            folder = root/'shift_data'/f'{target}_{suite}'
            if (folder/'arrays.npz').exists():
                if json.loads((folder/'config.json').read_text()) != signature:
                    raise ValueError('Shift cache configuration changed; use a new output directory')
                with np.load(folder/'arrays.npz') as a:
                    num, den, labels, groups = a['num'], a['den'], a['labels'], a['groups']
            else:
                nums, dens, labels, groups, metadata = [], [], [], [], []
                setting = replace(cfg, stress_suite=suite)
                for label, attack in enumerate(cfg.known_classes(target)):
                    for i in range(args.shift_runs):
                        seed = q.SeedBook(args.shift_seed).seed(f'cross_protocol_shift/{target}/{suite}/{attack}', i)
                        record, meta, _ = q.simulate_observations(setting, target, attack, seed, stress=True)
                        num, den, _ = q.window_features(target, record, cfg.windows)
                        nums.append(num); dens.append(den); labels.append(label)
                        groups.append(f'shift:{target}:{suite}:{attack}:{i}')
                        metadata.append(meta)
                num, den, labels, groups = np.asarray(nums), np.asarray(dens), np.asarray(labels), np.asarray(groups)
                folder.mkdir(parents=True, exist_ok=True)
                save_json(folder/'config.json', signature)
                save_json(folder/'metadata.json', metadata)
                np.savez_compressed(folder/'arrays.npz', num=num, den=den, labels=labels, groups=groups)
            shifted = Runs(target, num, den, labels, groups, {'test': np.arange(len(labels))}, list(q.FEATURES),
                           dict(**signature, interpretation='Controlled distribution change; strength/distance ranges may overlap training support. Not automatically unseen-parameter generalisation.'))
            for path in root.glob('*/*/*/*/seed_*/source_*/checkpoint.joblib'):
                payload = joblib.load(path)
                if payload['schema'] in ('full', 'specific') or target in payload['source_protocols']:
                    continue
                evaluate(payload, shifted, path.parent/f'isolated_{target}_{suite}', args, 'zero_shot')
