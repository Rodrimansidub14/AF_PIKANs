import unittest
import numpy as np
from src.sciml.signal_audit import localization


class LocalizationTests(unittest.TestCase):
    def test_perfect_localization(self):
        values=np.arange(1.,101.)
        r=localization(values,values)
        self.assertAlmostEqual(r['spearman'],1.)
        self.assertAlmostEqual(r['top_decile_overlap'],1.)
        self.assertAlmostEqual(r['error_enrichment'],95.5/50.5)

    def test_constant_signal_not_ranked_arbitrarily(self):
        r=localization(np.ones(100),np.arange(100.))
        self.assertIsNone(r['spearman'])
        self.assertIsNone(r['top_decile_overlap'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): localization(np.array([1.,np.nan]),np.ones(2))

