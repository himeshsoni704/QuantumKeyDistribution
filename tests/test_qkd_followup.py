"""Scientific-contract regressions; no performance claims from synthetic fixtures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from qkd_transfer.evaluation import metrics
from qkd_transfer.data import Runs
spec=importlib.util.spec_from_file_location('followup',Path(__file__).parents[1]/'tools/qkd_followup.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class FollowupTests(unittest.TestCase):
    def test_threshold_shift_does_not_change_ranking(self):
        y=np.array([0,0,1,1]);score=np.array([.7,.8,.85,.9])
        source=metrics(y,score,.5);adapted=metrics(y,score,.8)
        self.assertEqual(source['honest_false_alarm_rate'],1)
        self.assertEqual(adapted['honest_false_alarm_rate'],0)
        self.assertEqual(source['auroc'],adapted['auroc'])

    def test_fresh_schedule_has_unique_seeds_groups_and_honest_support(self):
        import qkd_pipeline as q
        args=f.parser().parse_args(['plan'])
        jobs=f.simulation_schedule(q.Config(),args,2026112101)
        other=f.simulation_schedule(q.Config(),args,2026112102)
        self.assertEqual(len(set(j[5] for j in jobs)),len(jobs))
        self.assertFalse(set(j[5] for j in jobs)&set(j[5] for j in other))
        self.assertEqual(len(set(j[6] for j in jobs+other)),len(jobs)+len(other))
        for protocol in ['bb84','bkm07','e91']:
            self.assertEqual(sum(j[0]==protocol and j[1]=='test' and j[2]==0 for j in jobs),300)
            self.assertEqual(sum(j[0]==protocol and j[1]=='cal_B' for j in jobs),200)

    def test_changed_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'contract.json';f.lock_manifest(path,{'seed':1})
            f.lock_manifest(path,{'seed':1})
            with self.assertRaisesRegex(ValueError,'changed'):f.lock_manifest(path,{'seed':2})

    def test_attack_rates_preserve_honest_and_unknown(self):
        result=f.attack_rows(np.array([0,0,1,-1]),np.array([.1,.9,.8,.7]),.5,['honest','pns'])
        rates={x['attack']:x for x in result}
        self.assertEqual(rates['honest']['rate'],.5)
        self.assertEqual(rates['pns']['rate'],1)
        self.assertEqual(rates['unknown_-1']['runs'],1)

    def test_calibration_rejects_test_overlap_before_scoring(self):
        data=Runs('e91',np.ones((2,1,1)),np.ones((2,1,1)),np.array([0,1]),np.array(['e91:a','e91:b']),
                  {'cal_B':np.array([0]),'test':np.array([0,1])},['qber_key'],{})
        payload={'train_groups':[],'validation_groups':[],'calibration':{'temperature_groups':[],'threshold_groups':[]}}
        with self.assertRaisesRegex(ValueError,'overlaps'):f.target_honest_calibration(payload,data,SimpleNamespace(alpha=.05))

    def test_target_threshold_ignores_test_and_calibration_attack_values(self):
        import sys
        sys.path.insert(0,str(Path(__file__).parent))
        from test_transfer_study import fixture
        from qkd_transfer.experiments import make_fit
        args=f.parser().parse_args(['review','--epochs','1','--bootstrap','0','--device','cpu'])
        opts=f.fit_options(args,'unused');opts.alpha=.5
        source,target=fixture('bkm07'),fixture('e91')
        payload=make_fit([source],'logistic','common','standard','raw',opts,0)
        before,groups=f.target_honest_calibration(payload,target,opts)
        attacked=target.splits['cal_B'][target.binary[target.splits['cal_B']]!=0]
        target.num[np.concatenate([attacked,target.splits['test']])]+=10000
        after,other=f.target_honest_calibration(payload,target,opts)
        self.assertEqual(before,after);self.assertEqual(groups,other)
        self.assertEqual(len(groups),4)

    def test_generation_resumes_without_resimulating_and_preserves_split_counts(self):
        import qkd_pipeline as q
        import json
        from dataclasses import asdict
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'original/data/bb84').mkdir(parents=True)
            (root/'original/data/bb84/config.json').write_text(json.dumps(asdict(q.Config())))
            args=f.parser().parse_args(['generate','--data-root',str(root/'original'),'--out',str(root/'new'),
                '--train-per-class','1','--val-per-class','1','--calibration-per-class','1',
                '--honest-calibration','1','--honest-test','2','--attack-test','1'])
            def simulate(cfg,protocol,attack,seed,stress):return {'seed':seed},{'seed':seed},{}
            def windows(protocol,record,count):
                rng=np.random.default_rng(record['seed'])
                shape=(count,len(q.FEATURES))
                return rng.uniform(0,1,shape),np.ones(shape),np.arange(count+1)
            with patch.object(q,'window_features',side_effect=windows):
                new=f.fresh_dataset(args,2026112101,simulator=simulate)
                before=f.sha256(new/'data/bb84/arrays.npz')
                with patch.object(q,'simulate_observations',side_effect=AssertionError('must resume')):
                    f.fresh_dataset(args,2026112101)
            self.assertEqual(before,f.sha256(new/'data/bb84/arrays.npz'))
            d=f.load_runs(new,'bb84')
            self.assertEqual(int((d.binary[d.splits['test']]==0).sum()),2)
            self.assertFalse(set(d.groups[d.splits['test']])&set(d.groups[d.splits['cal_B']]))

    def test_small_confirmation_execution(self):
        # Exercise real fits, paired output, every schema and export using fixtures, not simulator data.
        import sys
        sys.path.insert(0,str(Path(__file__).parent))
        from test_transfer_study import fixture
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            args=f.parser().parse_args(['confirm','--out',tmp,'--epochs','1','--bootstrap','0','--model-seeds','0','--device','cpu'])
            def bundle(root,p):
                d=fixture(p);d.provenance['config']={'classes':{p:['honest','intercept_resend']}}
                return d
            import torch
            torch.set_num_threads(1)
            with patch.object(f,'load_runs',side_effect=bundle):
                f.confirmation(args,'unused','test_confirmation')
            self.assertTrue((Path(tmp)/'test_confirmation/paired_gains.csv').exists())
            import pandas as pd
            gains=pd.read_csv(Path(tmp)/'test_confirmation/paired_gains.csv')
            self.assertEqual(set(gains.model),{'mlp','ae_lstm','jepa_lstm'})
            per=pd.read_csv(Path(tmp)/'test_confirmation/fresh_per_attack.csv')
            self.assertTrue({'qber','common','common_support','full','specific'}<=set(per.schema))

if __name__=='__main__':unittest.main()
