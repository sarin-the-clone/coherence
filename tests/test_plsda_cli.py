import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.plsda_discriminant import load_landmark_projection, main, score_figure


class PlsdaInputOptionsTest(unittest.TestCase):
    def test_missing_input_folder_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            errors = io.StringIO()
            with patch("sys.argv", ["plsda_discriminant.py", "--input-dir",
                                    str(root / "missing"), "--output-root",
                                    str(root / "outputs")]):
                with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Required PLS-DA descriptor export not found", errors.getvalue())
            self.assertFalse((root / "outputs").exists())

    def test_explicit_missing_landmark_file_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            errors = io.StringIO()
            with patch("sys.argv", ["plsda_discriminant.py", "--landmark-sc",
                                    str(root / "missing.json"), "--output-root",
                                    str(root / "outputs")]):
                with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Landmark semantic file not found", errors.getvalue())
            self.assertFalse((root / "outputs").exists())

    def test_landmark_projection_reads_the_selected_files(self):
        class IdentityProjection:
            @staticmethod
            def transform(values):
                return values

        features = ["W2k_off_diag_mean", "beta"]
        bins = [{"label": "reference", "range_tokens": [100_000, 150_000]}]
        stats = {"reference": (pd.Series(0.0, index=features),
                               pd.Series(1.0, index=features))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic = root / "selected_semantics.json"
            endpoint = root / "selected_endpoints.csv"
            semantic.write_text(json.dumps({
                "123": {"W2k": {"off_diag_mean": 0.4}},
            }), encoding="utf-8")
            pd.DataFrame([{
                "title": "Fixture", "n_tokens": 120_000,
                "file": "123_fixture.txt", "source_metric_file": "metrics_Frankenstein.json",
                "beta": 0.15,
            }]).to_csv(endpoint, index=False)
            result = load_landmark_projection(
                features, stats, IdentityProjection(), bins,
                sc_path=semantic, csv_path=endpoint,
            )
        np.testing.assert_allclose(result["xy"], [[0.4, 0.15]])
        self.assertEqual(result["labels"], ["F"])

    def test_biplot_renders_without_optional_landmarks(self):
        rng = np.random.default_rng(11)
        natural = rng.normal(size=(30, 2))
        generated = rng.normal(loc=0.5, size=(5, 2))
        bins = [{"label": "reference", "range_tokens": [100_000, 150_000]}]
        arrows = {
            "lv": np.array([[0.6, 0.4], [-0.4, 0.6]]),
            "labels": ["Fixture A", "Fixture B"],
            "colors": ["blue", "orange"],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "biplot.png"
            score_figure(
                output, natural, generated, np.full(5, "basic_conditioning"),
                np.full(30, "reference"), bins, 2, "combined", 0.5,
                arrows=arrows, landmarks=None,
            )
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
