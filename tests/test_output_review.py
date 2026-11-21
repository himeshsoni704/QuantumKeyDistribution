"""Regressions for saved-output joins, masking, labels and input preservation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location('review', Path(__file__).parents[1]/'tools/review_transfer_outputs.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReviewTests(unittest.TestCase):
    def test_report_masking_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; study=root/'study'; report=root/'report'
            write_json(data/'feature_schema.json', {'names':['qber_key']})
            (data/'bb84').mkdir()
            num=np.array([[[0.],[1.]],[[0.],[0.]],[[2.],[0.]],[[0.],[0.]]])
            den=np.array([[[1.],[3.]],[[0.],[0.]],[[2.],[2.]],[[4.],[4.]]])
            np.savez_compressed(data/'bb84/arrays.npz',num=num,den=den)
            pd.DataFrame({'group_id':['g0','g1','g2','g3'],'label':[0,1,2,0]}).to_csv(data/'bb84/runs.csv',index=False)
            digest=review.checksum(data/'bb84/arrays.npz')
            write_json(study/'study.json', {'datasets':{'bb84':{'arrays_sha256':digest,
                         'config':{'classes':{'bb84':['honest','intercept_resend','pns']}}}}})
            meta=dict(regime='zero_shot', source=['e91'], target='bb84', model='mlp',schema='common',
                      representation='raw',normalization='standard',seed=0,split='test',threshold=.5)
            folder=study/'common/standard/mlp/raw/seed_0/source_e91/zero_shot_bb84'
            write_json(folder/'metrics.json',meta)
            np.savez_compressed(folder/'predictions.npz',groups=['bb84:g0','bb84:g1','bb84:g2','bb84:g3'],
                                labels=[0,1,1,0],original_labels=[0,1,2,0],score=[.1,.9,.1,.8],threshold=.5)
            reference=review.qber_reference(data,'bb84',digest)
            self.assertAlmostEqual(reference.loc['bb84:g0','mean_window_qber'],1/6)
            self.assertAlmostEqual(reference.loc['bb84:g0','pooled_qber'],1/4)
            self.assertTrue(np.isnan(reference.loc['bb84:g1','pooled_qber']))
            args=['--study',str(study),'--data-root',str(data),'--out',str(report)]
            review.main(args)
            rates=pd.read_csv(report/'per_attack_by_seed.csv').set_index('attack')
            self.assertEqual(rates.loc['honest','rate'],.5)
            self.assertEqual(rates.loc['intercept_resend','rate'],1)
            self.assertEqual(rates.loc['pns','rate'],0)
            self.assertEqual(len(list((report/'plots').glob('*.png'))),1)
            self.assertEqual(review.checksum(data/'bb84/arrays.npz'),digest)
            with self.assertRaises(FileExistsError):
                review.main(args)
            with self.assertRaises(ValueError):
                review.qber_reference(data,'bb84','wrong')
            write_json(folder.parent/'within/metrics.json', {'threshold':.6,'regime':'within','model':'rf'})
            args[-1]=str(root/'other_report')
            with self.assertRaisesRegex(ValueError,'threshold differs'):
                review.main(args)

    def test_unknown_attack(self):
        frame=pd.DataFrame({'original_label':[0,-1],'alert':[False,True]})
        unknown=next(r for r in review.per_attack(frame,['honest','ir']) if r['original_label']==-1)
        self.assertEqual(unknown['attack'],'unknown_attack_-1')
        self.assertEqual(unknown['metric'],'attack_recall')

    def test_paired_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); seed=root/'common/standard/mlp/raw/seed_0'
            folder=seed/'source_e91/few_bb84_5_head_only'; out=root/'report'; out.mkdir()
            meta=dict(model='mlp',representation='raw',schema='common',normalization='standard',
                      source=['e91'],target='bb84',seed=0,metrics={'macro_f1':.6,'attack_recall':.5,
                      'honest_false_alarm_rate':0.,'auroc':.7})
            for path in (folder/'test',seed/'scratch_bb84_5/test'):
                write_json(path/'metrics.json',meta)
                np.savez_compressed(path/'predictions.npz',groups=['a','b'],labels=[0,1],
                                    original_labels=[0,1],score=[.1,.9],threshold=.5)
            write_json(folder/'budget.json', {'method':'head_only','budget_per_original_class':5})
            write_json(folder/'paired_transfer_gain.json', {'point':{k:0. for k in meta['metrics']},
                       'bootstrap':{k:{'ci95':[-.1,.1]} for k in meta['metrics']}})
            self.assertEqual(review.paired_comparisons(root,out,['mlp']),1)
            row=pd.read_csv(out/'paired_gains_by_seed.csv').iloc[0]
            self.assertEqual(row.transfer_attack_recall,.5)
            self.assertEqual(row.scratch_honest_false_alarm_rate,0)
            np.savez_compressed(seed/'scratch_bb84_5/test/predictions.npz',groups=['bad','b'],
                                labels=[0,1],original_labels=[0,1],score=[.1,.9],threshold=.5)
            with self.assertRaises(ValueError):
                review.paired_comparisons(root,out,['mlp'])


if __name__ == '__main__':
    unittest.main()
