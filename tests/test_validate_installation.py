import unittest

from utils.validate_installation import run_validation


class InstallationValidationTest(unittest.TestCase):
    def test_offline_descriptor_to_grouped_plsda_workflow(self):
        report = run_validation()

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["tokenizer"]["name"], "utf8_bytes")
        self.assertEqual(report["rows"], 32)
        self.assertEqual(report["independent_groups"], 16)
        self.assertEqual(len(report["folds"]), 3)
        self.assertEqual(sum(fold["test_rows"] for fold in report["folds"]), 32)
        self.assertEqual(sum(fold["test_groups"] for fold in report["folds"]), 16)
        self.assertIn("td_heaps_exponent", report["features"])
        self.assertIn("tr_repeat_gap_burstiness", report["features"])
        self.assertIn("tp_cond_entropy_4", report["features"])
        self.assertNotIn("tp_entropy_0", report["features"])
        for key in ("synthetic_group_auc", "synthetic_group_balanced_accuracy"):
            self.assertGreaterEqual(report[key], 0.0)
            self.assertLessEqual(report[key], 1.0)
