import unittest

from analysis.descriptor_schema import (
    CANONICAL_FEATURES,
    EXCLUDED_EXACT_ALIASES,
    EXCLUDED_MODEL_FEATURES,
    EXCLUDED_PROVENANCE_DUPLICATES,
    TOKEN_PREDICTABILITY_METRICS,
    TOKEN_RECURRENCE_METRICS,
    available_features,
)


class DescriptorSchemaTest(unittest.TestCase):
    def test_canonical_matrix_excludes_only_exact_aliases(self):
        self.assertEqual(len(CANONICAL_FEATURES), 44)
        self.assertEqual(len(TOKEN_PREDICTABILITY_METRICS), 13)
        self.assertIn("tr_repeat_gap_burstiness", TOKEN_RECURRENCE_METRICS)
        self.assertIn("tp_entropy_gain_1", TOKEN_PREDICTABILITY_METRICS)
        self.assertIn("tp_redundancy_4", TOKEN_PREDICTABILITY_METRICS)
        self.assertFalse(set(CANONICAL_FEATURES) & set(EXCLUDED_MODEL_FEATURES))
        self.assertIn("entropy_bits", CANONICAL_FEATURES)
        self.assertIn("perplexity", CANONICAL_FEATURES)
        self.assertNotIn("tp_entropy_0", CANONICAL_FEATURES)
        self.assertIn("tp_entropy_0", EXCLUDED_PROVENANCE_DUPLICATES)

    def test_available_features_preserves_canonical_order(self):
        columns = [*reversed(CANONICAL_FEATURES), *EXCLUDED_MODEL_FEATURES]
        self.assertEqual(available_features(columns), list(CANONICAL_FEATURES))
