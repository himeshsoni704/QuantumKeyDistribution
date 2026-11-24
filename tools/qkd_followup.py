#!/usr/bin/env python3
"""Saved-result controls, honest-only threshold adaptation and fresh-data confirmation.

Never alters legacy datasets/checkpoints. Synthetic tests are plumbing checks only.
"""
import argparse
from dataclasses import fields, asdict, replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import joblib
from qkd_transfer.data import Runs, load_runs, validate, label_budget, sha256
from qkd_transfer.evaluation import save_json, metrics, honest_threshold, temperature_score, bootstrap
from qkd_transfer.features import features
from qkd_transfer.experiments import cached_fit, make_fit, evaluate


def event(out, stage, **details):
    record = dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stage=stage, **details)
    print(json.dumps(record), flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    save_json(Path(out)/'status.json', record)
    with (Path(out)/'events.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')


def lock_manifest(path, value):
    path = Path(path)
    normalized = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError(f'Configuration/code/data changed: {path}; choose a new output directory')
    save_json(path, normalized)


def code_hashes():
    paths = [Path(__file__), ROOT/'qkd_pipeline.py'] + sorted((ROOT/'qkd_transfer').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def checked_payload(path):
    expected = json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    if sha256(path) != expected:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    return joblib.load(path)  # Only the user's trusted local checkpoints.


def target_honest_calibration(payload, data, args):
    ids = data.splits['cal_B']
    ids = ids[data.binary[ids] == 0]
    reference = set(data.groups[ids])
    used = set(payload['train_groups'] + payload['validation_groups'] +
               payload['calibration']['temperature_groups'] + payload['calibration']['threshold_groups'])
    if reference & (used | set(data.groups[data.splits['test']])):
        raise ValueError('Target honest reference overlaps source fitting or target test')
    if data.protocol in payload['source_protocols']:
        raise ValueError('Threshold adaptation requires a held-out target protocol')
    x, m, names = features(data, payload['schema'])
    if names != payload['normalizer'].names:
        raise ValueError('Incompatible feature coordinates')
    s = payload['model'].score(payload['normalizer'].transform(x[ids], m[ids]), m[ids], args)
    s = temperature_score(s, payload['calibration']['temperature'])
    return honest_threshold(s, args.alpha), sorted(reference)


def attack_rows(labels, score, threshold, names):
    result = []
    for label in np.unique(labels):
        take = labels == label
        hit = int((score[take] > threshold).sum())
        from qkd_transfer.evaluation import binomial_ci
        ci = binomial_ci(hit, int(take.sum()))
        result.append(dict(attack=names[int(label)] if 0 <= label < len(names) else f'unknown_{label}',
                           original_label=int(label), runs=int(take.sum()), alerts=hit,
                           rate=hit/int(take.sum()), ci95_low=ci[0], ci95_high=ci[1]))
    return result


def export_tables(out, rows, attack, stem):
    if rows:
        pd.DataFrame(rows).to_csv(out/f'{stem}_evaluations.csv', index=False)
    if attack:
        frame = pd.DataFrame(attack)
        frame.to_csv(out/f'{stem}_per_attack.csv', index=False)
        group = [k for k in ['stage','regime','source','target','model','schema','representation','normalization','method','budget','attack'] if k in frame]
        summary = frame.groupby(group, dropna=False).agg(mean_rate=('rate','mean'), sd_across_training_seeds=('rate','std'),
                                                        model_seed_rows=('rate','size'), runs_per_seed=('runs','first')).reset_index()
        summary.to_csv(out/f'{stem}_attack_summary.csv', index=False)


def saved_review(args, bundles):
    out = Path(args.out)/'saved_review'; out.mkdir(parents=True, exist_ok=True)
    rows, attacks, threshold_rows, adapted_attacks, skips = [], [], [], [], []
    studies = sorted(Path(args.existing_study).glob('*/study.json'))
    if not studies:
        raise ValueError('No saved study manifests found')
    cache = {}
    for manifest in studies:
        saved = json.loads(manifest.read_text());stage = manifest.parent.name
        for name, digest in saved['code'].items():
            current = ROOT/name if name == 'run_transfer_study.py' else ROOT/'qkd_transfer'/name
            if not current.exists() or sha256(current) != digest:
                raise ValueError(f'{stage}: fitted code differs: {name}')
        for protocol, d in bundles.items():
            if saved['datasets'][protocol]['arrays_sha256'] != d.provenance['arrays_sha256']:
                raise ValueError(f'{stage}: dataset differs for {protocol}')
        for f in sorted(manifest.parent.rglob('metrics.json')):
            meta = json.loads(f.read_text())
            if meta.get('split') != 'test' or meta.get('target') not in bundles:
                continue
            pred = f.parent/'predictions.npz'
            if not pred.exists():
                skips.append(dict(path=str(f), reason='missing predictions'));continue
            with np.load(pred, allow_pickle=False) as a:
                score, original, groups = a['score'], a['original_labels'], a['groups'].astype(str)
                threshold = float(a['threshold']); labels = a['labels']
            data = bundles[meta['target']];ids = data.splits['test']
            if not np.array_equal(groups, data.groups[ids]) or not np.array_equal(original, data.labels[ids]):
                raise ValueError(f'Prediction identities/labels differ: {pred}')
            if threshold != meta['threshold'] or not np.isfinite(score).all():
                raise ValueError(f'Invalid score/threshold: {pred}')
            ctx = dict(stage=stage, regime=meta['regime'], source='+'.join(meta['source']), target=meta['target'],
                       model=meta['model'], schema=meta['schema'], representation=meta['representation'],
                       normalization=meta['normalization'], seed=meta['seed'], method=meta['method'],
                       budget=meta.get('budget_per_original_class'), path=str(f.relative_to(Path(args.existing_study))))
            computed = metrics(labels, score, threshold)
            rows.append(dict(**ctx, threshold=threshold, **{k:v for k,v in computed.items() if not isinstance(v,(list,dict))}))
            names = data.provenance['config']['classes'][data.protocol]
            attacks.extend(dict(**ctx, **v) for v in attack_rows(original, score, threshold, names))
            if meta['regime'] != 'zero_shot':
                continue
            checkpoint = f.parent.parent/'checkpoint.joblib'
            if checkpoint not in cache:
                cache[checkpoint] = checked_payload(checkpoint)
            payload = cache[checkpoint]
            # Preserve source normalizer, model and temperature. Only the threshold changes.
            locked, ref = target_honest_calibration(payload, data, fit_options(args, out))
            adjusted = metrics(labels, score, locked['threshold'])
            threshold_rows.append(dict(**ctx, source_threshold=threshold, target_honest_threshold=locked['threshold'],
                                       target_honest_calibration_runs=len(ref), auroc=computed['auroc'],
                                       source_recall=computed['attack_recall'], target_honest_recall=adjusted['attack_recall'],
                                       source_far=computed['honest_false_alarm_rate'], target_honest_far=adjusted['honest_false_alarm_rate'],
                                       interpretation='Target-honest threshold adaptation; not strict zero-shot'))
            adapted_attacks.extend(dict(**ctx, **v) for v in attack_rows(original, score, locked['threshold'], names))
        cache.clear()
        event(args.out, 'saved_review_stage', study=stage, evaluations=len(rows))
    export_tables(out, rows, attacks, 'all_saved')
    pd.DataFrame(threshold_rows).to_csv(out/'threshold_adaptation.csv', index=False)
    pd.DataFrame(adapted_attacks).to_csv(out/'threshold_adaptation_per_attack.csv', index=False)
    save_json(out/'coverage.json', dict(studies=[str(p) for p in studies], evaluations=len(rows), threshold_comparisons=len(threshold_rows), skipped=skips))
    draw_review(out)


def draw_review(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(out/'all_saved_per_attack.csv')
    bb = frame[(frame.target == 'bb84') & (frame.regime == 'within')]
    if len(bb):
        mean = bb.groupby(['stage','schema','representation','normalization','model','attack']).rate.mean().unstack('attack')
        mean.to_csv(out/'bb84_within_all_controls.csv')
        for stage, part in mean.groupby(level='stage'):
            labels = [' | '.join(map(str, idx[1:])) for idx in part.index]
            fig, ax = plt.subplots(figsize=(11,max(4,len(part)*.28)))
            im=ax.imshow(part.values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
            ax.set_xticks(range(len(part.columns)),part.columns,rotation=25,ha='right');ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set_title(f'BB84 within-protocol controls: {stage}\nHonest column = false alarms; attack columns = recall')
            fig.colorbar(im,ax=ax,label='Rate (0-1)');fig.tight_layout();fig.savefig(out/f'bb84_{stage}.png',dpi=150);plt.close(fig)
    shift = pd.read_csv(out/'threshold_adaptation.csv')
    primary=shift[(shift.stage=='primary') & shift.model.isin(['logistic','mlp','ae_lstm','jepa_lstm'])]
    if len(primary):
        summary=primary.groupby(['source','target'])[['source_far','target_honest_far','source_recall','target_honest_recall','auroc']].mean()
        summary.to_csv(out/'threshold_direction_summary.csv')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,cols,title in [(axes[0],['source_far','target_honest_far'],'Honest false alarms'),(axes[1],['source_recall','target_honest_recall'],'Attack recall')]:
            summary[cols].plot.bar(ax=ax);ax.set_ylim(0,1.05);ax.set_title(title);ax.set_xlabel('Source, target');ax.legend(['Source threshold','Target-honest threshold'])
        fig.suptitle('Threshold adaptation uses target calibration honest runs only; model and ranking are unchanged')
        fig.tight_layout();fig.savefig(out/'threshold_adaptation.png',dpi=150);plt.close(fig)


def simulation_schedule(cfg, args, data_seed):
    import qkd_pipeline as q
    jobs=[]
    for protocol in cfg.protocols:
        for split in ['train','val','cal_A','cal_B','test']:
            for label, attack in enumerate(cfg.known_classes(protocol)):
                count = {'train':args.train_per_class,'val':args.val_per_class,'cal_A':args.calibration_per_class,
                         'cal_B':args.honest_calibration if label==0 else 0,
                         'test':args.honest_test if label==0 else args.attack_test}[split]
                for i in range(count):
                    seed=q.SeedBook(data_seed).seed(f'confirmation_v1/{protocol}/{split}/{attack}',i)
                    group=f'fresh{data_seed}_{protocol}_{split}_{label}_{i:06d}'
                    jobs.append((protocol,split,label,attack,i,seed,group))
    return jobs


def protocols_for(args):
    return ('bkm07', 'e91') if args.focused else ('bb84', 'bkm07', 'e91')


def fresh_dataset(args, data_seed, simulator=None):
    import qkd_pipeline as q
    saved=json.loads((Path(args.data_root)/'data/bb84/config.json').read_text())
    cfg=q.Config(**{k:v for k,v in saved.items() if k in {f.name for f in fields(q.Config)}})
    folder=Path(args.out)/f'fresh_data_{data_seed}'; folder.mkdir(parents=True,exist_ok=True)
    cfg=replace(cfg,out=str(folder),data_seed=data_seed,protocols=protocols_for(args))
    signature=dict(data_seed=data_seed,simulator_sha256=sha256(q.__file__),configuration=asdict(cfg),
                   split_sizes={k:getattr(args,k) for k in ['train_per_class','val_per_class','calibration_per_class','honest_calibration','honest_test','attack_test']})
    lock_manifest(folder/'generation_contract.json',signature)
    save_json(folder/'data/feature_schema.json',dict(names=q.FEATURES))
    jobs=simulation_schedule(cfg,args,data_seed)
    simulate=simulator or q.simulate_observations
    for protocol in cfg.protocols:
        dest=folder/'data'/protocol;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():
            load_runs(folder,protocol) # Validates published arrays and metadata before reuse.
            continue
        nums=[];dens=[];rows=[];meta=[];splits={k:[] for k in ['train','val','cal_A','cal_B','test']}
        selected=[j for j in jobs if j[0]==protocol]
        for index,(_,split,label,attack,i,seed,group) in enumerate(selected):
            cache=dest/'features'/f'{group}.npz';cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                with np.load(cache,allow_pickle=False) as a:
                    if int(a['seed'])!=seed or str(a['simulator_sha256'])!=signature['simulator_sha256']:
                        raise ValueError(f'Invalid simulation cache {cache}')
                    num,den,details=a['num'],a['den'],json.loads(str(a['metadata']))
                    if not np.isfinite(num).all() or not np.isfinite(den).all() or (den<0).any():
                        raise ValueError(f'Corrupt simulation statistics: {cache}')
            else:
                record,details,_=simulate(cfg,protocol,attack,seed,False)
                num,den,_=q.window_features(protocol,record,cfg.windows)
                del record
                tmp=cache.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,num=num,den=den,seed=seed,simulator_sha256=signature['simulator_sha256'],metadata=json.dumps(q.json_safe(details)))
                tmp.replace(cache)
            nums.append(num);dens.append(den);splits[split].append(index)
            rows.append(dict(group_id=group,run_id=group,label=label,seed=seed,scenario_group=split))
            meta.append(dict(group_id=group,**details))
            if index%10==0 or index+1==len(selected):
                event(args.out,'generating',data_seed=data_seed,protocol=protocol,completed=index+1,total=len(selected))
        arrays=dest/'arrays.npz';tmp=dest/'arrays.tmp.npz'
        np.savez_compressed(tmp,num=np.asarray(nums),den=np.asarray(dens));tmp.replace(arrays)
        pd.DataFrame(rows).to_csv(dest/'runs.csv',index=False)
        save_json(dest/'splits.json',splits)
        save_json(dest/'config.json',dict(**asdict(cfg),source_sha256=signature['simulator_sha256']))
        save_json(dest/'eval_meta.json',meta)
        save_json(dest/'audit_manifest.json',dict(arrays_sha256=sha256(arrays),run_count=len(rows)))
        load_runs(folder,protocol)
        save_json(dest/'complete.json',dict(runs=len(rows),contract=signature))
    return folder


def fit_options(args, out):
    return SimpleNamespace(out=str(out), epochs=args.epochs,patience=10,batch_size=32,lr=.001,device=args.device,
                           threads=args.threads,alpha=.05,bootstrap=args.bootstrap)


def confirmation(args, data_root, tag):
    out=Path(args.out)/tag;out.mkdir(parents=True,exist_ok=True)
    bundles={p:load_runs(data_root,p) for p in protocols_for(args)}
    lock_manifest(out/'contract.json',dict(code=code_hashes(),datasets={p:d.provenance for p,d in bundles.items()},
                  model_seeds=args.model_seeds,budget_per_original_class=5,epochs=args.epochs,bootstrap=args.bootstrap,
                  primary='BKM07 -> E91 JEPA head-only minus matched JEPA scratch macro-F1',
                  note='Historical repeat is exploratory; fresh datasets are the preselected replication. Shared validation/calibration costs are additional.'))
    opts=fit_options(args,out);gains=[];results=[];result_paths=[];attack=[]
    source,target=bundles['bkm07'],bundles['e91']
    for seed in args.model_seeds:
        chosen=label_budget(target,5,seed)
        for model in ['logistic','rf','mlp','ae_lstm','jepa_lstm','qber']:
            schema,norm=('qber','raw') if model=='qber' else ('common','standard')
            representation='latent' if model in ['ae_lstm','jepa_lstm'] else 'raw'
            base=out/'confirmation'/model/f'seed_{seed}'
            scratch=cached_fit(base/'scratch',lambda:make_fit([target],model,schema,norm,representation,opts,seed,selected=chosen,budget=5))
            r=evaluate(scratch,target,base/'scratch/test',opts,'few_shot');results.append(r);result_paths.append(base/'scratch/test')
            names=target.provenance['config']['classes']['e91']
            with np.load(base/'scratch/test/predictions.npz',allow_pickle=False) as a:
                sy,ss,sg,sth=a['labels'],a['score'],a['groups'],float(a['threshold'])
            if model not in ['mlp','ae_lstm','jepa_lstm']:continue
            original=cached_fit(base/'source_bkm07',lambda:make_fit([source],model,schema,norm,representation,opts,seed))
            zero=evaluate(original,target,base/'source_bkm07/zero_shot_e91',opts,'zero_shot')
            results.append(zero);result_paths.append(base/'source_bkm07/zero_shot_e91')
            locked,reference=target_honest_calibration(original,target,opts)
            with np.load(base/'source_bkm07/zero_shot_e91/predictions.npz',allow_pickle=False) as z:
                adjusted=metrics(z['labels'],z['score'],locked['threshold'])
            save_json(base/'target_honest_threshold.json',dict(calibration=locked,reference_groups=reference,metrics=adjusted,source_threshold=zero['threshold'],interpretation='Target-honest threshold adaptation, not strict zero-shot'))
            adapted=cached_fit(base/'head_only',lambda:make_fit([target],model,schema,norm,representation,opts,seed,selected=chosen,initial=original,mode='head_only',budget=5))
            r=evaluate(adapted,target,base/'head_only/test',opts,'few_shot');results.append(r);result_paths.append(base/'head_only/test')
            with np.load(base/'head_only/test/predictions.npz',allow_pickle=False) as a:
                if not np.array_equal(a['groups'],sg) or not np.array_equal(a['labels'],sy):raise ValueError('Unpaired comparison')
                gain={k:r['metrics'][k]-metrics(sy,ss,sth)[k] for k in ['macro_f1','attack_recall','honest_false_alarm_rate','auroc']}
                ci=bootstrap(sy,a['score'],sg,float(a['threshold']),args.bootstrap,seed,ss,sth)
            gains.append(dict(model=model,seed=seed,dataset=tag,**gain))
            save_json(base/'paired_gain.json',dict(point=gain,bootstrap=ci,selected_target_groups=target.groups[chosen].tolist(),label_usage=r['label_usage']))
            event(args.out,'confirmation_fit',dataset=tag,model=model,seed=seed,gain=gain)
    # Matched low-label controls above; full-budget BB84 feature controls below.
    if 'bb84' in bundles:
        bb=bundles['bb84']
        for schema in ['qber','common','common_support','full','specific']:
            for model in (['qber'] if schema=='qber' else ['logistic','rf','mlp','ae_lstm','jepa_lstm']):
                for seed in args.model_seeds:
                    norm='raw' if model=='qber' else 'standard'
                    for representation in (['latent','combined'] if model in ['ae_lstm','jepa_lstm'] else ['raw']):
                        base=out/'bb84_controls'/schema/model/representation/f'seed_{seed}'
                        payload=cached_fit(base,lambda:make_fit([bb],model,schema,norm,representation,opts,seed))
                        r=evaluate(payload,bb,base/'test',opts,'within');results.append(r);result_paths.append(base/'test')
    for r, directory in zip(results, result_paths):
        ctx={k:r[k] for k in ['regime','target','model','schema','representation','normalization','seed','method']}
        ctx.update(source='+'.join(r['source']),budget=r.get('budget_per_original_class'))
        with np.load(directory/'predictions.npz',allow_pickle=False) as a:
            names=bundles[r['target']].provenance['config']['classes'][r['target']]
            attack.extend(dict(**ctx,**x) for x in attack_rows(a['original_labels'],a['score'],float(a['threshold']),names))
    flat=[]
    for r in results:
        ctx={k:r[k] for k in ['regime','target','model','schema','representation','normalization','seed','method']}
        flat.append(dict(**ctx,**{k:v for k,v in r['metrics'].items() if not isinstance(v,(list,dict))}))
    export_tables(out,flat,attack,'fresh' if tag!='historical_confirmation' else 'historical')
    pd.DataFrame(gains).to_csv(out/'paired_gains.csv',index=False)
    event(args.out,'dataset_complete',dataset=tag)


def aggregate_replications(args):
    allrows=[]
    for seed in args.data_seeds:
        f=Path(args.out)/f'replication_{seed}/paired_gains.csv'
        if not f.exists():continue
        frame=pd.read_csv(f);frame['data_seed']=seed;allrows.append(frame)
    if not allrows:return
    frame=pd.concat(allrows,ignore_index=True);frame.to_csv(Path(args.out)/'replication_all_gains.csv',index=False)
    # Data-generation seed is the replication unit; do not pool model seeds as independent datasets.
    bydata=frame.groupby(['model','data_seed'])[['macro_f1','attack_recall','honest_false_alarm_rate','auroc']].mean().reset_index()
    bydata.to_csv(Path(args.out)/'replication_dataset_means.csv',index=False)
    summary=bydata.groupby('model').agg(datasets=('data_seed','size'),mean_f1_gain=('macro_f1','mean'),
                                      sd_across_datasets=('macro_f1','std'),positive_datasets=('macro_f1',lambda s:int((s>0).sum())))
    summary.to_csv(Path(args.out)/'replication_summary.csv')
    if args.focused:
        rng = np.random.default_rng(0)
        intervals = []
        for model, rows in bydata.groupby('model'):
            columns = ['macro_f1', 'attack_recall', 'honest_false_alarm_rate', 'auroc']
            values = rows[columns].to_numpy()
            count = len(values)
            bounds = np.full((2, len(columns)), np.nan)
            if count >= 3 and args.bootstrap:
                indices = rng.integers(count, size=(args.bootstrap, count))
                bounds = np.percentile(values[indices].mean(axis=1), [2.5, 97.5], axis=0)
            for metric, mean, lo, hi in zip(columns, values.mean(axis=0), *bounds):
                intervals.append(dict(model=model, metric=metric, datasets=count, mean_gain=mean,
                                      paired_ci95_low=lo, paired_ci95_high=hi))
        pd.DataFrame(intervals).to_csv(Path(args.out)/'replication_paired_intervals.csv', index=False)
    save_json(Path(args.out)/'replication_interpretation.json',dict(primary='JEPA BKM07 -> E91 head-only minus scratch F1',
               independent_unit='fresh data-generation seed; training seeds averaged within dataset',
               limitations='Three datasets are limited replication evidence, not a strong population-level significance result. No automatic publication claim.'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','review','confirm','generate','replicate','all'])
    p.add_argument('--focused', action='store_true',
                   help='Generate and train BKM07 and E91 only; omit BB84 controls.')
    p.add_argument('--data-root',default='out_full_new');p.add_argument('--existing-study',default='out_transfer_final')
    p.add_argument('--out',default='out_followup_20261121')
    p.add_argument('--data-seeds',nargs='+',type=int,default=[2026112101,2026112102,2026112103])
    p.add_argument('--model-seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--train-per-class',type=int,default=210);p.add_argument('--val-per-class',type=int,default=30)
    p.add_argument('--calibration-per-class',type=int,default=50);p.add_argument('--honest-calibration',type=int,default=200)
    p.add_argument('--honest-test',type=int,default=300);p.add_argument('--attack-test',type=int,default=100)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    for k in ['train_per_class','val_per_class','calibration_per_class','honest_calibration','honest_test','attack_test','epochs','threads']:
        if getattr(args,k)<1:raise ValueError(f'{k} must be positive')
    if args.epochs>100 or args.bootstrap<0 or len(set(args.data_seeds))!=len(args.data_seeds) or len(set(args.model_seeds))!=len(args.model_seeds):raise ValueError('Invalid epochs/bootstrap or duplicate seeds')
    if 20260913 in args.data_seeds:raise ValueError('Fresh data seeds must differ from the historical seed')
    if args.focused and args.command not in ['plan', 'generate', 'replicate']:
        raise ValueError('--focused requires plan, generate or replicate')
    out=Path(args.out).resolve()
    for source in [Path(args.data_root).resolve(),Path(args.existing_study).resolve()]:
        if out==source or out in source.parents or source in out.parents:raise ValueError('Use a separate sibling output directory')
    if args.command=='plan':
        counts={}
        classes = {'bb84': 5, 'bkm07': 3, 'e91': 4}
        for protocol in protocols_for(args):
            nclass = classes[protocol]
            counts[protocol]=nclass*(args.train_per_class+args.val_per_class+args.calibration_per_class)+args.honest_calibration+args.honest_test+(nclass-1)*args.attack_test
        print(json.dumps(dict(data_seeds=args.data_seeds,simulations_per_dataset=counts,total_simulations=sum(counts.values())*len(args.data_seeds),
                             honest_test_per_protocol_per_dataset=args.honest_test,epochs_cap=args.epochs,source_data_untouched=True),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    lock_manifest(out/'followup_contract.json',dict(code=code_hashes(),args={k:v for k,v in vars(args).items() if k not in ['command','device','threads'] and (k != 'focused' or v)}))
    torch.set_num_threads(args.threads)
    try:
        if args.command in ['review','all']:
            event(out,'review_started');saved_review(args,{p:load_runs(args.data_root,p) for p in ['bb84','bkm07','e91']})
        if args.command in ['confirm','all']:
            event(out,'historical_confirmation_started');confirmation(args,args.data_root,'historical_confirmation')
        if args.command in ['generate','replicate','all']:
            for seed in args.data_seeds:
                root=fresh_dataset(args,seed)
                if args.command in ['replicate', 'all']:
                    confirmation(args, root, f'replication_{seed}')
                    aggregate_replications(args)
        event(out,'complete',command=args.command)
    except Exception as exc:
        event(out,'failed',error=str(exc));raise


if __name__=='__main__':main()
