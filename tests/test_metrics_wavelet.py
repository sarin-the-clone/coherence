import unittest

import numpy as np

from metrics import CoherenceMetrics, wavelet_dfa


class WaveletDfaParityTest(unittest.TestCase):
    def test_cpu_fallback_matches_established_torch_path(self):
        tokens = np.random.default_rng(7).integers(0, 1_000, size=4_096)

        expected = CoherenceMetrics.wavelet_dfa_torch(tokens, device="cpu")
        actual = wavelet_dfa(tokens, device="cpu")

        np.testing.assert_allclose(actual[:3], expected[:3], rtol=0, atol=0)
        np.testing.assert_array_equal(actual[3]["scales"], expected[3]["scales"])
        np.testing.assert_allclose(
            actual[3]["fluctuation"], expected[3]["fluctuation"], rtol=0, atol=0
        )
