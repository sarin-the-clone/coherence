import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analysis.descriptor_schema import (
    CANONICAL_FEATURES, SC_FEATURES, STAT_INFO_METRICS,
    TOKEN_DISTRIBUTION_METRICS, TOKEN_RECURRENCE_METRICS, TOKEN_PREDICTABILITY_METRICS,
)
from analysis.plsda_inputs import (
    GENERATED_DESCRIPTOR_EXPORT, NATURAL_PREDICTABILITY_EXPORT,
    NATURAL_SEMANTIC_EXPORT, NATURAL_TOKEN_EXPORT, load_export_data,
)


class PlsdaExportInputTest(unittest.TestCase):
    bins = [
        {"label": "100-150k", "range_tokens": [100_000, 150_000]},
        {"label": ">150k", "range_tokens": [150_000, float("inf")]},
    ]

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def prepare(self, root):
        semantic = pd.DataFrame([
            {"source": "natural", "id": book_id, "label": f"Book {book_id}",
             "bin": bin_id, "n_tokens": count,
             **{feature: 0.12345678901234567 for feature in SC_FEATURES}}
            for book_id, bin_id, count in ((2, 1, 180_000), (1, 0, 120_000),
                                            (3, 0, 130_000), (4, 0, 140_000))
        ])
        natural_path = root / NATURAL_SEMANTIC_EXPORT
        natural_path.parent.mkdir(parents=True, exist_ok=True)
        semantic.to_csv(natural_path, index=False)
        self.write_json(natural_path.with_name("sc_pca_summary.json"),
                        {"standardization_mode": "none"})

        batch = [{"book_id": book_id, **{feature: float(book_id)
                  for feature in STAT_INFO_METRICS}} for book_id in (1, 2, 3, 4)]
        self.write_json(root / "batch_results" / "batch_001.json", batch)
        token_features = (*TOKEN_DISTRIBUTION_METRICS, *TOKEN_RECURRENCE_METRICS)
        self.write_jsonl(root / NATURAL_TOKEN_EXPORT, [
            {"book_id": book_id, **{feature: float(book_id) for feature in token_features},
             "td_zipf_slope": -7.0} for book_id in (1, 2, 3, 4)
        ])
        # Book 3 has an incomplete descriptor; book 4 has no predictability record.
        self.write_jsonl(root / NATURAL_PREDICTABILITY_EXPORT, [
            {"book_id": book_id,
             **{feature: (None if book_id == 3 and feature == "tp_cond_entropy_1"
                          else float(book_id)) for feature in TOKEN_PREDICTABILITY_METRICS},
             "tp_entropy_0": 999.0} for book_id in (1, 2, 3)
        ])
        generated = pd.DataFrame([
            {"run_id": "fixture-run", "conditioning_mode": "basic_conditioning",
             "bin": bin_id, "n_tokens": count,
             **{feature: 5.0 for feature in CANONICAL_FEATURES},
             "PC1": 999.0, "tp_entropy_0": 999.0}
            for bin_id, count in ((0, 95_000), (1, 180_000))
        ])
        generated_path = root / GENERATED_DESCRIPTOR_EXPORT
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(generated_path, index=False)
        return semantic, generated

    def test_joins_saved_exports_with_canonical_schema_and_original_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "custom-input-folder"
            self.prepare(root)
            natural, generated, features = load_export_data(root, self.bins)

            self.assertEqual(features, list(CANONICAL_FEATURES))
            self.assertEqual(natural["book_id"].tolist(), [2, 1])
            self.assertEqual(natural["n_tokens"].tolist(), [180_000, 120_000])
            self.assertEqual(natural["size_bin"].tolist(), [">150k", "100-150k"])
            self.assertEqual(generated["size_bin"].tolist(), ["100-150k", ">150k"])
            self.assertEqual(natural.loc[0, "W2k_off_diag_mean"], 0.12345678901234567)
            self.assertEqual(generated["run_id"].tolist(), ["fixture-run", "fixture-run"])
            for excluded in ("PC1", "tp_entropy_0", "td_zipf_slope"):
                self.assertNotIn(excluded, natural.columns)
                self.assertNotIn(excluded, generated.columns)

    def test_rejects_ambiguous_book_and_run_prefix_rows(self):
        for duplicate_source in ("natural", "generated"):
            with self.subTest(source=duplicate_source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                semantic, generated = self.prepare(root)
                if duplicate_source == "natural":
                    pd.concat([semantic, semantic.iloc[[0]]]).to_csv(
                        root / NATURAL_SEMANTIC_EXPORT, index=False)
                else:
                    pd.concat([generated, generated.iloc[[0]]]).to_csv(
                        root / GENERATED_DESCRIPTOR_EXPORT, index=False)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    load_export_data(root, self.bins)

    def test_rejects_scaled_semantic_input_and_missing_canonical_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, generated = self.prepare(root)
            summary = (root / NATURAL_SEMANTIC_EXPORT).with_name("sc_pca_summary.json")
            self.write_json(summary, {"standardization_mode": "bin"})
            with self.assertRaisesRegex(ValueError, "raw semantic values are required"):
                load_export_data(root, self.bins)
            self.write_json(summary, {"standardization_mode": "none"})
            generated.drop(columns=["tp_cond_entropy_4"]).to_csv(
                root / GENERATED_DESCRIPTOR_EXPORT, index=False)
            with self.assertRaisesRegex(ValueError, "tp_cond_entropy_4"):
                load_export_data(root, self.bins)

    def test_missing_input_error_identifies_persistent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing-inputs"
            with self.assertRaisesRegex(FileNotFoundError, "sc_pca_features_robust_z.csv"):
                load_export_data(root, self.bins)
