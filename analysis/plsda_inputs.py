"""Load the saved, raw descriptor exports used by PLS-DA.

All paths are relative to a persistent input directory, normally
``natural_texts/Output``. Loading these measurements requires neither model
inference nor the original temporary JSON files and generation chunk records.
The natural-book joins and complete-case selection match the original loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.descriptor_schema import (
    CANONICAL_FEATURES, SC_FEATURES, STAT_INFO_METRICS, TOKEN_METRICS,
)
from analysis.pca_legacy_size_bins import assign_bin


NATURAL_SEMANTIC_EXPORT = Path("pca/sc_raw/sc_pca_features_robust_z.csv")
GENERATED_DESCRIPTOR_EXPORT = Path("pca/sc_combined/sc_combined_pca_generated_scores.csv")
NATURAL_TOKEN_EXPORT = Path("gutenberg_benchmark_cache/token_naturalness_metrics.jsonl")
NATURAL_PREDICTABILITY_EXPORT = Path("gutenberg_benchmark_cache/token_predictability_metrics.jsonl")


def _require_columns(frame: pd.DataFrame, columns, path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required PLS-DA descriptor export not found: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required PLS-DA descriptor cache not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return pd.DataFrame(records)


def _unique(frame: pd.DataFrame, keys: list[str], path: Path) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, keys].head(5).to_dict("records")
        raise ValueError(f"{path}: duplicate {', '.join(keys)} records: {examples}")


def _book_source(frame: pd.DataFrame, path: Path,
                 book_ids: pd.Series) -> pd.DataFrame:
    _require_columns(frame, ["book_id"], path)
    frame = frame.copy()
    frame["book_id"] = pd.to_numeric(frame["book_id"], errors="coerce").astype("Int64")
    # Records outside the semantic subcorpus never enter the original inner join.
    frame = frame.loc[frame["book_id"].isin(book_ids)].copy()
    _unique(frame, ["book_id"], path)
    return frame


def _complete_rows(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in [*features, "n_tokens"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (frame.replace([np.inf, -np.inf], np.nan)
                 .dropna(subset=[*features, "n_tokens"]).reset_index(drop=True))


def load_export_data(input_dir: Path, size_bins: list[dict]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return matched-size natural/generated rows and the canonical 44 columns.

    Natural semantic rows retain their measured token counts and source order.
    Inner joins and complete-case filtering are applied before feature-set
    ablations, as in the original analysis. Generated rows retain their run/bin
    identities; saved PCA scores and diagnostic aliases are not model inputs.
    """
    input_dir = Path(input_dir)
    natural_path = input_dir / NATURAL_SEMANTIC_EXPORT
    summary_path = natural_path.with_name("sc_pca_summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mode = summary.get("standardization_mode")
        if mode is not None and mode != "none":
            raise ValueError(f"{summary_path}: raw semantic values are required; "
                             f"standardization_mode is {mode!r}, not 'none'")

    semantic = _read_csv(natural_path)
    _require_columns(semantic, ["source", "id", "n_tokens", *SC_FEATURES], natural_path)
    semantic = semantic.loc[semantic["source"] == "natural"].copy()
    semantic["book_id"] = pd.to_numeric(semantic["id"], errors="raise").astype("Int64")
    if semantic["book_id"].isna().any():
        raise ValueError(f"{natural_path}: natural rows must have a book ID")
    _unique(semantic, ["book_id"], natural_path)
    semantic["title"] = semantic["label"] if "label" in semantic.columns else ""
    if "bin" not in semantic.columns:
        semantic["bin"] = "?"
    natural = semantic[["book_id", "title", "bin", "n_tokens", *SC_FEATURES]].copy()
    if natural.empty:
        raise ValueError(f"{natural_path}: no natural-book semantic rows found")

    batch_dir = input_dir / "batch_results"
    batch_paths = sorted(batch_dir.glob("batch_*.json"))
    if not batch_paths:
        raise FileNotFoundError(f"Required PLS-DA batch_*.json records not found: {batch_dir}")
    records = []
    for path in batch_paths:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    batch = _book_source(pd.DataFrame(records), batch_dir, natural["book_id"])
    _require_columns(batch, STAT_INFO_METRICS, batch_dir)

    token_path = input_dir / NATURAL_TOKEN_EXPORT
    predictability_path = input_dir / NATURAL_PREDICTABILITY_EXPORT
    token = _book_source(_read_jsonl(token_path), token_path, natural["book_id"])
    predictability = _book_source(_read_jsonl(predictability_path), predictability_path,
                                  natural["book_id"])
    token_columns = [column for column in TOKEN_METRICS if column in token.columns]
    predictability_columns = [column for column in TOKEN_METRICS
                             if column in predictability.columns]
    overlap = set(token_columns) & set(predictability_columns)
    if overlap:
        raise ValueError("Ambiguous natural token descriptors occur in both caches: "
                         + ", ".join(sorted(overlap)))
    natural = (natural.merge(batch[["book_id", *STAT_INFO_METRICS]], on="book_id", how="inner")
                      .merge(token[["book_id", *token_columns]], on="book_id", how="inner")
                      .merge(predictability[["book_id", *predictability_columns]],
                             on="book_id", how="inner"))
    features = list(CANONICAL_FEATURES)
    _require_columns(natural, features, input_dir)
    natural = _complete_rows(natural, features)
    natural["size_bin"] = [assign_bin(count, size_bins) for count in natural["n_tokens"]]
    natural = natural.dropna(subset=["size_bin"]).reset_index(drop=True)

    generated_path = input_dir / GENERATED_DESCRIPTOR_EXPORT
    generated = _read_csv(generated_path)
    _require_columns(generated, ["run_id", "conditioning_mode", "bin", "n_tokens", *features],
                     generated_path)
    if generated[["run_id", "bin"]].isna().any().any():
        raise ValueError(f"{generated_path}: generated rows must identify both run_id and bin")
    _unique(generated, ["run_id", "bin"], generated_path)
    generated = generated[["run_id", "conditioning_mode", "bin", "n_tokens", *features]].copy()
    generated = _complete_rows(generated, features)
    generated["size_bin"] = [assign_bin(count, size_bins, lower_edge_tolerance=0.10)
                             for count in generated["n_tokens"]]
    generated = generated.dropna(subset=["size_bin"]).reset_index(drop=True)
    if natural.empty or generated.empty:
        raise ValueError("No complete, size-matched PLS-DA rows remain for one of the two classes")
    return natural, generated, features
