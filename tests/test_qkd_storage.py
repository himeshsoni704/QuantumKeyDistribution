"""Storage-cache tests; these verify format/integrity, not QKD physics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qkd_pipeline import FEATURES, _atomic_savez_compressed, _load_feature_record


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.feature = self.root / "features" / "bb84_000000.npz"
        self.original = self.root / "original_features" / "bb84_000000.json"
        self.num = np.arange(4 * len(FEATURES), dtype=np.float64).reshape(4, len(FEATURES))
        self.den = np.full_like(self.num, 10.0)
        self.edges = np.arange(5, dtype=np.int64)
        self.meta = {"distance": 25.0, "attack_strength": 0.2}
        self.audit = {"qber": 0.05, "chsh_S": None}
        self.config_digest = "config-test"
        self.source_sha256 = "source-test"
        self.original.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_record(self):
        _atomic_savez_compressed(
            self.feature,
            num=self.num,
            den=self.den,
            edges=self.edges,
            run_id=np.asarray("bb84_000000"),
            protocol=np.asarray("bb84"),
            seed=np.asarray(123, dtype=np.int64),
            windows=np.asarray(4, dtype=np.int64),
            config_digest=np.asarray(self.config_digest),
            source_sha256=np.asarray(self.source_sha256),
            meta_json=np.asarray(json.dumps(self.meta, sort_keys=True)),
        )
        self.original.write_text(json.dumps(self.audit))

    def load(self, config_digest=None, source_sha256=None):
        return _load_feature_record(
            self.feature,
            self.original,
            "bb84_000000",
            "bb84",
            123,
            4,
            config_digest or self.config_digest,
            source_sha256 or self.source_sha256,
        )

    def test_atomic_feature_record_round_trip(self):
        self.write_record()
        loaded = self.load()
        self.assertIsNotNone(loaded)
        num, den, edges, meta, audit = loaded
        np.testing.assert_array_equal(num, self.num)
        np.testing.assert_array_equal(den, self.den)
        np.testing.assert_array_equal(edges, self.edges)
        self.assertEqual(meta, self.meta)
        self.assertEqual(audit, self.audit)
        self.assertFalse(list(self.feature.parent.glob("*.tmp")))

    def test_cache_rejects_changed_provenance(self):
        self.write_record()
        self.assertIsNone(self.load(config_digest="different-config"))
        self.assertIsNone(self.load(source_sha256="different-source"))

    def test_corrupt_archive_is_treated_as_cache_miss(self):
        self.write_record()
        self.feature.write_bytes(b"not a valid npz archive")
        self.assertIsNone(self.load())


if __name__ == "__main__":
    unittest.main()
