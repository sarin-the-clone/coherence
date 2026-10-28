import csv
import io
from pathlib import Path
import tempfile
import unittest

from analysis.descriptor_schema import CANONICAL_FEATURES, SC_FEATURES
from utils.check_reproduction_inputs import check_inputs


class ReproductionInputPresenceTest(unittest.TestCase):
    @staticmethod
    def put(path, content="{}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def put_csv(self, path, row):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        self.put(path, stream.getvalue())

    def prepare(self, input_dir):
        for relative in (
            "batch_results/batch_1.json",
            "gutenberg_benchmark_cache/token_naturalness_metrics.jsonl",
            "gutenberg_benchmark_cache/token_predictability_metrics.jsonl",
            "analysis/size_binned_metrics.json",
        ):
            self.put(input_dir / relative)
        self.put_csv(input_dir / "pca/sc_raw/sc_pca_features_robust_z.csv", {
            "source": "natural", "id": "12", "n_tokens": 121_000,
            **dict.fromkeys(SC_FEATURES, 0.5),
        })
        table = input_dir / "pca/sc_combined/sc_combined_pca_generated_scores.csv"
        self.put_csv(table, {
            "run_id": "fixture-run", "conditioning_mode": "basic_conditioning",
            "bin": "0", "n_tokens": 121_000,
            **dict.fromkeys(CANONICAL_FEATURES, 0.5),
        })
        return table

    def test_persistent_exports_need_no_temporary_files_or_chunk_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            input_dir = root / "natural_texts/Output"
            self.prepare(input_dir)
            report = check_inputs(root=root)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["generated_runs_referenced"], 1)
            self.assertEqual(report["input_directory"], str(input_dir))

    def test_input_directory_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            input_dir = Path(directory) / "persistent-data"
            self.prepare(input_dir)
            size_stats = Path(directory) / "custom-size-bins.json"
            self.put(size_stats)
            report = check_inputs(root=root, input_dir=input_dir, size_stats=size_stats)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["input_directory"], str(input_dir))
            size_check = next(item for item in report["checks"] if item["name"] == "size_bin_statistics")
            self.assertEqual(size_check["paths"], [str(size_stats)])

    def test_missing_size_stats_reports_optional_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            self.prepare(input_dir)
            (input_dir / "analysis/size_binned_metrics.json").unlink()
            report = check_inputs(input_dir=input_dir)
            self.assertEqual(report["missing"], [])
            check = next(item for item in report["checks"] if item["name"] == "size_bin_statistics")
            self.assertFalse(check["present"])
            self.assertFalse(check["required"])
            self.assertIn("built-in size bins", check["detail"])

    def test_export_header_and_data_rows_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            table = self.prepare(input_dir)
            valid_header = table.read_text(encoding="utf-8").splitlines()[0] + "\n"
            for content in ("", "run_id\nfixture-run\n", valid_header):
                self.put(table, content)
                self.assertIn("generated_descriptor_export", check_inputs(input_dir=input_dir)["missing"])

    def test_landmark_overrides_make_endpoint_csv_required(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "data"
            self.prepare(input_dir)
            landmark = Path(directory) / "optional-landmark.json"
            endpoint = Path(directory) / "optional-endpoints.csv"
            self.assertEqual(check_inputs(input_dir=input_dir, landmark_sc=landmark)["missing"],
                             ["landmark_semantic_table"])
            self.put(landmark)
            report = check_inputs(input_dir=input_dir, landmark_sc=landmark, landmark_csv=endpoint)
            self.assertIn("landmark_endpoint_descriptors", report["missing"])
            self.put(endpoint)
            self.assertEqual(check_inputs(input_dir=input_dir, landmark_sc=landmark,
                                          landmark_csv=endpoint)["missing"], [])
