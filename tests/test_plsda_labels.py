import unittest

from analysis.plsda_discriminant import descriptor_label


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
