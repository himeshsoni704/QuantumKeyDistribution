"""Small integration checks; no full experimental generation."""
import tempfile
import unittest
import numpy as np
import qkd_pipeline as q

class ResearchChecks(unittest.TestCase):
    def test_e91_optics_exact_budget(self):
        f = q.extract_e91_features(target_k_key_pairs=20, n_windows=5,
            detector={'optical_efficiency': .3, 'dark_count': .001},
            return_record=True, rng=np.random.default_rng(2))
        self.assertEqual(f['_k_achieved'], 20)
        self.assertTrue(0 < f['_detection_yield'] < .2)
        record = dict(zip(('a','b','ra','rb'), f['_record']))
        num, den, _ = q.window_features('e91', record, 5)
        self.assertEqual(den[:,q.FEATURE_INDEX['qber_key']].sum(), 20)

    def test_jepa_train_calibrate(self):
        for representation in ('jepa', 'hybrid'):
            q.torch.set_num_threads(2)
            with tempfile.TemporaryDirectory() as out:
                c = q.Config(out=out, protocols=('bb84',), classes={'bb84':['honest','intercept_resend']},
                    windows=8, epochs=2, folds=2, feature_set=representation, seeds=(0,))
                rng=np.random.default_rng(3)
                den=np.ones((40,8,len(q.FEATURES)))*20
                num=rng.binomial(20,.1,size=den.shape).astype(float)
                y=np.repeat([0,1],20); groups=np.array([str(i) for i in range(40)])
                splits=q.split_groups(y,groups,0); tr=splits['train']
                norm=q.Normalizer.fit(num[tr],den[tr],groups[tr])
                data=q.DataBundle(num,den,np.zeros((40,9)),y,groups,splits,norm.transform(num,den),norm)
                model=q.train_model(c,'bb84',representation + '_lstm',0,bundles={'bb84':data})
                locked=q.lock_calibration(c,'bb84',representation + '_lstm',0,model,data)
                self.assertEqual(locked['anomaly_available'], representation == 'hybrid')
                result=q.predict(model,data,splits['test'],'bb84',c)
                self.assertTrue(np.isfinite(result['logits']).all())

    def test_bkm_optical_paths(self):
        b=q.simulate_bkm07_batch(100000, 0, 'none', 0, 0,
            rng=np.random.default_rng(5), p_meas=0, p_prep=0, p_ret=0,
            detector=dict(optical_model='two_way', eta_bob_detector=.5,
                eta_alice_detector=.4, reflection_efficiency=1,
                resend_efficiency=1, bob_dark=0, alice_dark=0))
        ctrl=np.isin(b['round_type'], ['CTRL_Z','CTRL_X'])
        sift=~ctrl
        self.assertAlmostEqual(b['survived'][ctrl].mean(), .4, delta=.01)
        self.assertAlmostEqual(b['survived'][sift].mean(), .2, delta=.01)
        key=b['survived'] & (b['round_type']=='SIFT_KEY')
        self.assertTrue(np.all(b['bit_B'][key] == b['bit_A_final'][key]))

    def test_dark_and_dead_time(self):
        clicks, signal=q.detector_events(np.zeros(10,bool), np.random.default_rng(1),
                                        dark_probability=1, dead_slots=2)
        self.assertEqual(np.flatnonzero(clicks).tolist(), [0,3,6,9])
        self.assertFalse(signal.any())

    def test_confidence_interval(self):
        low, high=q.binomial_interval(0,30)
        self.assertEqual(low,0)
        self.assertGreater(high,.05)

    def test_transfer_end_to_end(self):
        import contextlib, io, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            cfg=q.Config(out=out, protocols=('bb84','bkm07'),
                classes={'bb84':['honest','intercept_resend'], 'bkm07':['honest','symmetric_attack']},
                runs_per_class=20, stress_per_class=2, calibration_extra_per_class=2,
                key_events=4, windows=4, epochs=1, folds=2, seeds=(0,),
                label_budgets=(5,), smoke=True, hardware_effects=True, feature_set='hybrid')
            q.generate(cfg)
            q.transfer_experiments(cfg)
            rows=json.loads((Path(out)/'tables/transfer_hybrid_lstm.json').read_text())
            self.assertEqual(len(rows),12)
            self.assertEqual({r['method'] for r in rows}, {'scratch','head_only','partial'})
            self.assertTrue(all('honest_false_alert_ci95' in r for r in rows))

if __name__ == '__main__':
    unittest.main()
