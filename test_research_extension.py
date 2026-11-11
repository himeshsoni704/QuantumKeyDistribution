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
        q.torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as out:
            c = q.Config(out=out, protocols=('bb84',), classes={'bb84':['honest','intercept_resend']},
                windows=8, epochs=2, folds=2, feature_set='jepa', seeds=(0,))
            rng=np.random.default_rng(3)
            den=np.ones((40,8,len(q.FEATURES)))*20
            num=rng.binomial(20,.1,size=den.shape).astype(float)
            y=np.repeat([0,1],20); groups=np.array([str(i) for i in range(40)])
            splits=q.split_groups(y,groups,0); tr=splits['train']
            norm=q.Normalizer.fit(num[tr],den[tr],groups[tr])
            data=q.DataBundle(num,den,np.zeros((40,9)),y,groups,splits,norm.transform(num,den),norm)
            model=q.train_model(c,'bb84','jepa_lstm',0,bundles={'bb84':data})
            locked=q.lock_calibration(c,'bb84','jepa_lstm',0,model,data)
            self.assertFalse(locked['anomaly_available'])
            result=q.predict(model,data,splits['test'],'bb84',c)
            self.assertTrue(np.isfinite(result['logits']).all())

if __name__ == '__main__':
    unittest.main()
