"""Read-only presence/header check for the persistent PLS-DA input exports.

Run ``python -m utils.check_reproduction_inputs`` from the repository root.
Only standard-library modules are used. This does not load models, download
data, validate descriptors or establish reproduction of the manuscript results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.descriptor_schema import CANONICAL_FEATURES, SC_FEATURES


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "natural_texts" / "Output"


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def check_inputs(root: Path | None = None, input_dir: Path | None = None,
                 size_stats: Path | None = None, landmark_sc: Path | None = None,
                 landmark_csv: Path | None = None) -> dict:
    """Inspect persistent loader inputs without importing scientific libraries."""
    landmark_requested = landmark_sc is not None
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    input_dir = Path(input_dir) if input_dir is not None else root / "natural_texts" / "Output"
    size_stats = Path(size_stats) if size_stats is not None else input_dir / "analysis" / "size_binned_metrics.json"
    landmark_sc = Path(landmark_sc) if landmark_sc is not None else input_dir / "pca" / "generated_prefixes" / "landmark_sc.json"
    landmark_csv = Path(landmark_csv) if landmark_csv is not None else input_dir / "pca" / "generated_prefixes" / "landmark_prefix_projection_scores.csv"
    checks = []

    def add(name, paths, required=True, detail=""):
        paths = list(paths)
        present = [path for path in paths if _nonempty(path)]
        check = {
            "name": name, "required": required, "present": bool(present),
            "paths": [str(path) for path in paths],
            "nonempty_files": len(present), "detail": detail,
        }
        checks.append(check)
        return check

    def add_csv(name, path, required_columns):
        check = add(name, [path])
        if not check["present"]:
            return
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                missing_columns = sorted(set(required_columns) - set(reader.fieldnames or []))
                if missing_columns:
                    check["present"] = False
                    check["detail"] = "Missing CSV columns: " + ", ".join(missing_columns)
                    return
                rows = list(reader)
            check["rows"] = len(rows)
            if not rows:
                check["present"] = False
                check["detail"] = "CSV has a header but no data rows."
            elif name == "generated_descriptor_export":
                check["generated_runs_referenced"] = len({row["run_id"] for row in rows})
            else:
                check["natural_rows"] = sum(row["source"] == "natural" for row in rows)
                if not check["natural_rows"]:
                    check["present"] = False
                    check["detail"] = "CSV contains no source=natural rows."
        except (OSError, UnicodeError, csv.Error) as error:
            check["present"] = False
            check["detail"] = f"Cannot read CSV: {error}"

    add_csv("natural_semantic_export",
            input_dir / "pca" / "sc_raw" / "sc_pca_features_robust_z.csv",
            ("source", "id", "n_tokens", *SC_FEATURES))
    add_csv("generated_descriptor_export",
            input_dir / "pca" / "sc_combined" / "sc_combined_pca_generated_scores.csv",
            ("run_id", "conditioning_mode", "bin", "n_tokens", *CANONICAL_FEATURES))
    batch_dir = input_dir / "batch_results"
    add("natural_batch_results", sorted(batch_dir.glob("batch_*.json")),
        detail=f"At least one nonempty batch_*.json in {batch_dir}")
    for name, filename in (
        ("natural_distribution_recurrence", "token_naturalness_metrics.jsonl"),
        ("natural_predictability", "token_predictability_metrics.jsonl"),
    ):
        add(name, [input_dir / "gutenberg_benchmark_cache" / filename])
    add("size_bin_statistics", [size_stats],
        required=False,
        detail="When absent, the loader uses built-in size bins. Compare the "
               "original bin definitions before claiming exact reproduction.")
    add("landmark_semantic_table", [landmark_sc], required=landmark_requested)
    add("landmark_endpoint_descriptors",
        [landmark_csv],
        required=landmark_sc.exists(),
        detail="Required whenever landmark_sc.json exists: the analysis then reads this CSV.")
    missing = [check for check in checks if check["required"] and not check["present"]]
    return {
        "purpose": "presence_check_not_scientific_validation",
        "status": "missing_inputs" if missing else "presence_checks_passed",
        "repository_root": str(root),
        "input_directory": str(input_dir),
        "scope": "Checks nonempty input files, export CSV headers and row counts. "
                 "Does not validate descriptor values, complete book/run coverage, "
                 "prefix selection, model versions, or agreement with manuscript results. "
                 "PLS-DA uses these persistent exports without temporary calculation files "
                 "or original generated chunk records.",
        "generated_runs_referenced": next((check.get("generated_runs_referenced", 0)
            for check in checks if check["name"] == "generated_descriptor_export"), 0),
        "checks": checks,
        "missing": [check["name"] for check in missing],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Persistent input root (default: natural_texts/Output)")
    parser.add_argument("--size-stats", type=Path,
                        help="Optional override for the size-bin definitions")
    parser.add_argument("--landmark-sc", type=Path,
                        help="Optional override for the landmark semantic JSON")
    parser.add_argument("--landmark-csv", type=Path,
                        help="Optional override for the landmark endpoint CSV")
    args = parser.parse_args()
    report = check_inputs(input_dir=args.input_dir, size_stats=args.size_stats,
                          landmark_sc=args.landmark_sc, landmark_csv=args.landmark_csv)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["missing"] else 0)


if __name__ == "__main__":
    main()
