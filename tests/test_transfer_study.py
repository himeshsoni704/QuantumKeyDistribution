"""Fast synthetic plumbing tests, not evidence of QKD detection performance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from qkd_transfer.data import Runs, validate, label_budget, load_runs, sha256
from qkd_transfer.features import features, Preprocessor
from qkd_transfer.models import BinaryModel, seed_all
from qkd_transfer.experiments import make_fit, evaluate, run_study
from qkd_transfer.evaluation import honest_threshold, metrics, bootstrap
from qkd_transfer.analysis import diagnose
from run_transfer_study import parser, plan


def fixture(protocol):
    rng = np.random.default_rng({'bb84': 1, 'bkm07': 2, 'e91': 3}[protocol])
    y = np.tile([0, 1], 30)
    den = np.ones((60, 5, 3))*40
    prob = np.where(y == 0, .03, .2)[:, None, None]
    num = rng.binomial(40, np.broadcast_to(prob, den.shape)).astype(float)
    groups = np.asarray([f'{protocol}:{i}' for i in range(60)])
    splits = {'train': np.arange(0, 24), 'val': np.arange(24, 32), 'cal_A': np.arange(32, 40), 'cal_B': np.arange(40, 48), 'test': np.arange(48, 60), 'stress': np.array([], dtype=int)}
    return validate(Runs(protocol, num, den, y, groups, splits,
                         ['qber_key', 'qber_zsr', 'chsh_S'], {'fixture': True}))


def options(out='unused'):
    a = parser().parse_args(['run', '--data-root', '/unused', '--out', str(out)])
    a.models = ['logistic', 'mlp', 'ae_lstm', 'jepa_lstm']
    a.schemas = ['common']
    a.normalizations = ['standard']
    a.representations = ['latent']
    a.seeds = [0]
    a.budgets = [0, 2]
    a.epochs = 2
    a.patience = 1
    a.bootstrap = 2
    a.device = 'cpu'
    a.batch_size = 8
    a.alpha = .2
    a.explain_rows = 12
    a.threads = 1
    return a


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_group_independence(self):
        d = fixture('bb84')
        d.groups[24] = d.groups[0]
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate(d)

    def test_unknown_attack_binary_label(self):
        d = fixture('bb84')
        d.labels[-1] = -1
        self.assertEqual(d.binary[-1], 1)

    def test_feature_schema_and_duplicate_removal(self):
        d = fixture('bkm07')
        x, m, names = features(d, 'common')
        self.assertEqual(x.shape, (60, 5, 3))
        self.assertFalse(m[:, 0, 1:].any())
        self.assertNotIn('qber_zsr', features(d, 'full')[2])
        self.assertNotIn('qber_key', features(d, 'specific')[2])

    def test_normalizer_only_source(self):
        d = fixture('bb84')
        x, m, names = features(d, 'common')
        ids = d.splits['train']
        norm = Preprocessor.fit(x[ids], m[ids], d.groups[ids], names)
        before = norm.mean.copy()
        norm.transform(x*100, m)
        np.testing.assert_array_equal(norm.mean, before)
        self.assertEqual(set(norm.fit_groups), set(d.groups[ids]))

    def test_budget_determinism_and_zero_rejection(self):
        d = fixture('bb84')
        np.testing.assert_array_equal(label_budget(d, 2, 7), label_budget(d, 2, 7))
        self.assertEqual(len(label_budget(d, 2, 7)), 4)
        with self.assertRaises(ValueError):
            label_budget(d, 0, 0)
        with self.assertRaises(ValueError):
            label_budget(d, 100, 0)

    def test_calibration_and_metric_edges(self):
        self.assertTrue(honest_threshold(np.zeros(3), .05)['insufficient_resolution'])
        result = metrics(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), .5)
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['honest_false_alarm_rate'], 0)
        self.assertEqual(result['auroc'], 1)
        r = bootstrap([0, 0, 1, 1], [.1, .2, .8, .9], ['a','a','b','b'], .5, 10, 0,
                      [.1, .2, .8, .9], .5)
        self.assertEqual(r['macro_f1']['ci95'], [0, 0])

    def test_target_labels_do_not_change_source_or_predictions(self):
        a = options()
        source, target = fixture('bb84'), fixture('e91')
        fitted = make_fit([source], 'mlp', 'common', 'standard', 'raw', a, 4)
        state = copy.deepcopy(fitted['model'].net.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            evaluate(fitted, target, Path(tmp)/'first', a, 'zero_shot')
            target.labels = 1-target.labels
            evaluate(fitted, target, Path(tmp)/'second', a, 'zero_shot')
            with np.load(Path(tmp)/'first/predictions.npz') as x, np.load(Path(tmp)/'second/predictions.npz') as y:
                np.testing.assert_array_equal(x['score'], y['score'])
        for k, v in state.items():
            self.assertTrue(torch.equal(v, fitted['model'].net.state_dict()[k]))
        contaminated = copy.deepcopy(fitted)
        contaminated['normalizer'].fit_groups.append(target.groups[0])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            evaluate(contaminated, target, tmp, a, 'zero_shot')

    def test_head_only_freezes_source_mapping(self):
        a = options()
        source, target = fixture('bb84'), fixture('bkm07')
        initial = make_fit([source], 'ae_lstm', 'common', 'standard', 'latent', a, 0)
        adapted = make_fit([target], 'ae_lstm', 'common', 'standard', 'latent', a, 0,
                           selected=label_budget(target, 2, 0), initial=initial, mode='head_only')
        for name, value in initial['model'].net.state_dict().items():
            if not name.startswith('head.'):
                self.assertTrue(torch.equal(value, adapted['model'].net.state_dict()[name]), name)
        self.assertEqual(adapted['normalizer'].fit_groups, initial['normalizer'].fit_groups)
        self.assertEqual(adapted['model'].history['trainable_parameters'], ['head.weight', 'head.bias'])

    def test_all_regimes_resume_checkpoint_and_diagnostics(self):
        bundles = {p: fixture(p) for p in ('bb84', 'bkm07', 'e91')}
        with tempfile.TemporaryDirectory() as tmp:
            a = options(Path(tmp)/'study')
            a.models = ['mlp', 'jepa_lstm']
            run_study(a, bundles)
            paths = list(Path(a.out).rglob('checkpoint.joblib'))
            before = {str(p): sha256(p) for p in paths}
            a.resume = True
            run_study(a, bundles)
            self.assertEqual(before, {str(p): sha256(p) for p in paths})
            self.assertTrue((Path(a.out)/'transfer_gaps.csv').exists())
            self.assertTrue((Path(a.out)/'label_efficiency.csv').exists())
            lopo = list(Path(a.out).glob('*/standard/mlp/raw/seed_0/lopo_*/test/metrics.json'))
            self.assertEqual(len(lopo), 3)
            for path in lopo:
                record = json.loads(path.read_text())
                self.assertNotIn(record['target'], record['source'])
            import joblib
            source = next(Path(a.out).glob('*/standard/mlp/raw/seed_0/source_bb84/checkpoint.joblib'))
            diagnose(joblib.load(source), bundles, source.parent/'diagnostics', a)
            self.assertTrue((source.parent/'diagnostics/attack_probes.json').exists())
            a.lr = .02
            with self.assertRaises(ValueError):
                run_study(a, bundles)

    def test_read_only_historical_loader(self):
        d = fixture('bb84')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'bb84'; path.mkdir()
            (root/'feature_schema.json').write_text(json.dumps({'names': d.names}))
            (path/'config.json').write_text(json.dumps({'source_sha256': 'historical-version'}))
            np.savez_compressed(path/'arrays.npz', num=d.num, den=d.den)
            pd.DataFrame({'label': d.labels, 'group_id': d.groups}).to_csv(path/'runs.csv', index=False)
            (path/'splits.json').write_text(json.dumps({k: v.tolist() for k,v in d.splits.items()}))
            digest = sha256(path/'arrays.npz')
            loaded = load_runs(root, 'bb84')
            self.assertEqual(loaded.provenance['simulator_source_sha256'], 'historical-version')
            self.assertFalse(loaded.provenance['manifest_present'])
            self.assertEqual(digest, sha256(path/'arrays.npz'))
            (path/'audit_manifest.json').write_text(json.dumps({'arrays_sha256': 'wrong', 'run_count': 60}))
            with self.assertRaises(ValueError):
                load_runs(root, 'bb84')


if __name__ == '__main__':
    unittest.main()
