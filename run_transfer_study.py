#!/usr/bin/env python3
"""Read-only dataset reuse, frozen-source transfer and explicit target adaptation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from qkd_transfer.data import load_runs
from qkd_transfer.evaluation import save_json, clean
from qkd_transfer.experiments import run_study, study_fingerprint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'run', 'analyze', 'shifts', 'report'])
    p.add_argument('--data-root', required=True, help='Existing out/data or existing out directory; never modified')
    p.add_argument('--out', default='out_transfer_v1')
    p.add_argument('--protocols', nargs='+', default=['bb84', 'bkm07', 'e91'])
    p.add_argument('--models', nargs='+', choices=['qber', 'logistic', 'rf', 'boosting', 'mlp', 'lstm', 'ae_lstm', 'jepa_lstm'], default=['qber', 'logistic', 'rf', 'boosting', 'mlp', 'ae_lstm', 'jepa_lstm'])
    p.add_argument('--schemas', nargs='+', choices=['qber', 'common', 'common_support', 'full', 'specific'], default=['qber', 'common'])
    p.add_argument('--normalizations', nargs='+', choices=['raw', 'standard', 'honest'], default=['raw', 'standard'])
    p.add_argument('--representations', nargs='+', choices=['latent', 'combined'], default=['latent', 'combined'])
    p.add_argument('--regimes', nargs='+', choices=['within', 'zero_shot', 'few_shot', 'lopo', 'joint', 'reference'], default=['within', 'zero_shot', 'few_shot', 'lopo', 'joint'])
    p.add_argument('--budgets', nargs='+', type=int, default=[0, 5, 20, 100, -1])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--shap', action='store_true')
    p.add_argument('--explain-rows', type=int, default=128)
    p.add_argument('--analysis-limit', type=int, default=0, help='0 means all source checkpoints')
    p.add_argument('--shift-runs', type=int, default=30)
    p.add_argument('--shift-seed', type=int, default=20261112)
    p.add_argument('--shift-suites', nargs='+', choices=['drift','strength','bursty','visibility','distance'], default=['drift','strength','bursty','visibility','distance'])
    p.add_argument('--allow-new-simulator', action='store_true')
    return p


def plan(args, bundles):
    jobs = 0
    for schema in args.schemas:
        for norm in args.normalizations:
            for model in args.models:
                if model == 'qber' and (schema != 'qber' or norm != 'raw'):
                    continue
                nrepr = len(args.representations) if model in ('ae_lstm', 'jepa_lstm') else 1
                n = len(bundles)
                count = n  # One source fit reused across all targets.
                if 'joint' in args.regimes:
                    count += 1
                if schema not in ('full', 'specific'):
                    if 'lopo' in args.regimes and n >= 3:
                        count += n
                if 'few_shot' in args.regimes and model in ('mlp','lstm','ae_lstm','jepa_lstm'):
                    for d in bundles.values():
                        smallest = min((d.labels[d.splits['train']] == c).sum() for c in set(d.labels[d.splits['train']]))
                        available = sum(b != 0 and (b < 0 or b <= smallest) for b in args.budgets)
                        count += available*(1+2*(n-1))
                jobs += count*nrepr*len(args.seeds)
    array_bytes = sum(d.num.nbytes+d.den.nbytes for d in bundles.values())
    return dict(training_jobs=jobs, max_epochs_per_training_phase=args.epochs,
                early_stopping_patience=args.patience, input_arrays_ram_gib=array_bytes/2**30,
                estimated_artifact_upper_gib=jobs*8/1024,
                storage_estimate_note='Conservative 8 MiB/job allowance for compact checkpoints/predictions/JSON; optional SHAP and shifts add storage. No raw dataset copies.',
                runtime='Measure one representative source fit on your machine. Total depends on early stopping, CPU baselines, bootstrap and number of AE/JEPA phases; no unsupported hour estimate.',
                cases={p: {'runs': len(d.labels), 'split_counts': {k: len(v) for k,v in d.splits.items()},
                           'historical_simulator': d.provenance.get('simulator_source_sha256'),
                           'cross_split_duplicate_observations': d.provenance.get('cross_split_duplicate_observations')} for p,d in bundles.items()})


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.epochs <= 100 or args.patience < 1 or args.bootstrap < 0 or args.batch_size < 1:
        raise ValueError('Require 1..100 epochs, positive patience/batch size and nonnegative bootstrap count')
    if any(b < -1 for b in args.budgets) or len(set(args.protocols)) != len(args.protocols):
        raise ValueError('Invalid budgets or duplicate protocols')
    if not 0 < args.alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    data_root, out = Path(args.data_root).resolve(), Path(args.out).resolve()
    if out == data_root or out in data_root.parents or data_root in out.parents:
        raise ValueError('Use a separate sibling output directory, outside the existing data/output tree')
    torch.set_num_threads(args.threads)
    bundles = {p: load_runs(args.data_root, p) for p in args.protocols}
    overview = plan(args, bundles)
    print(json.dumps(clean(overview), indent=2), flush=True)
    if args.command == 'plan':
        return
    if args.command == 'run':
        run_study(args, bundles)
        return
    saved = json.loads((out/'study.json').read_text())
    current = study_fingerprint(args, bundles)
    if saved['code'] != current['code'] or saved['datasets'] != current['datasets']:
        raise ValueError('Analysis input code/dataset differs from fitted study')
    if args.command == 'report':
        from qkd_transfer.analysis import summarize
        summarize(out)
    elif args.command == 'analyze':
        import joblib
        from qkd_transfer.analysis import diagnose
        paths = sorted(out.glob('*/*/*/*/seed_*/source_*/checkpoint.joblib'))
        if args.analysis_limit:
            paths = paths[:args.analysis_limit]
        for path in paths:
            print('Diagnose', path, flush=True)
            diagnose(joblib.load(path), bundles, path.parent/'diagnostics', args)
    elif args.command == 'shifts':
        from qkd_transfer.shifts import run_shifts
        run_shifts(args, bundles)


if __name__ == '__main__':
    main()
