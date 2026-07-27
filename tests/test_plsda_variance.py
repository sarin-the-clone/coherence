import unittest

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from analysis.plsda_discriminant import pls_component_x_variance


class PlsdaVarianceTest(unittest.TestCase):
    def test_component_variance_is_incremental_and_bounded(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(120, 6))
        y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=len(X))
        pls = PLSRegression(n_components=4).fit(X, y)

        variance = pls_component_x_variance(pls, X)

        self.assertEqual(variance.shape, (4,))
        self.assertTrue(np.all(variance >= -1e-10))
        self.assertLessEqual(float(variance.sum()), 100.0 + 1e-10)
