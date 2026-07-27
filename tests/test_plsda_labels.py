import unittest

import numpy as np

from analysis.plsda_discriminant import aggregate_group_scores, descriptor_label


class PlsdaLabelTest(unittest.TestCase):
    def test_biplot_and_vip_share_descriptor_labels(self):
        self.assertEqual(
            descriptor_label("W2k_off_diag_mean"),
            "Global similarity (2k window)",
        )
        self.assertEqual(
            descriptor_label("tr_repeat_gap_burstiness"),
            "Repeat-gap burstiness",
        )
        self.assertEqual(
            descriptor_label("tp_entropy_gain_1"),
            "Entropy gain (order 1)",
        )

    def test_group_score_aggregation_collapses_nested_prefixes(self):
        targets, scores = aggregate_group_scores(
            np.array([1, 1, 1, 0, 0]),
            np.array([0.9, 0.7, 0.8, 0.1, 0.2]),
            np.array(["run-1", "run-1", "run-1", "book-1", "book-2"]),
        )

        np.testing.assert_array_equal(targets, np.array([1, 0, 0]))
        np.testing.assert_allclose(scores, np.array([0.8, 0.1, 0.2]))
