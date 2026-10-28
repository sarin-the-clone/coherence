"""Offline CPU smoke check for token descriptors and grouped PLS-DA.

Run from the repository root with ``python -m utils.validate_installation``.
The texts and labels are synthetic software fixtures. The reported scores are
installation diagnostics, not estimates of human/generated-text separability.
No language model, embedding model, external corpus or network is required.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path


def run_validation(seed: int = 7) -> dict:
    """Exercise the real estimators, fold-local scaling and run-level scoring."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold

    from analysis.descriptor_schema import available_features
    from analysis.plsda_discriminant import aggregate_group_scores, vip_scores
    from analysis.pca_legacy_size_bins import fit_per_bin_stats, standardize_per_bin
    from metrics import (
        shannon_entropy,
        token_distribution_scaling_metrics,
        token_predictability_metrics,
        token_recurrence_metrics,
    )
    from tokenizer import Tokenizer

    rng = np.random.default_rng(seed)
    tokenizer = Tokenizer(kind="bytes", add_space_punct=False)
    torch.set_num_threads(1)

    def describe(tokens: list[int]) -> dict:
        entropy, perplexity = shannon_entropy(tokens, device="cpu")
        if not np.isclose(perplexity, 2.0 ** entropy, rtol=1e-6):
            raise AssertionError("Perplexity must equal 2**entropy_bits")
        return {
            "entropy_bits": entropy,
            "perplexity": perplexity,
            **token_distribution_scaling_metrics(tokens),
            **token_recurrence_metrics(tokens),
            **token_predictability_metrics(tokens, max_order=4, device="cpu"),
        }

    # Each reference fixture is one independent text. Its shuffled counterpart
    # has three nested prefixes that must remain together in cross-validation.
    # These labels describe the fixtures, not human or language-model authorship.
    records = []
    for index, place in enumerate((
        "harbor", "forest", "garden", "station", "library", "village", "island", "valley",
    )):
        text = (
            f"At the {place}, a visitor opened a notebook and described the morning. "
            "One page held a question; the next returned to a detail noticed earlier. "
            f"The account followed {index + 2} paths before returning to its beginning. "
            f"A quiet conversation about the {place} continued after the journey. "
        ) * (18 + index)
        tokens = tokenizer.encode(text)
        if tokenizer.encode(tokenizer.decode(tokens)) != tokens:
            raise AssertionError("Byte tokenizer round trip changed the fixture")
        records.append({
            **describe(tokens), "target": 0, "group": f"reference-{index}",
            "n_tokens": len(tokens), "size_bin": "synthetic-smoke-bin",
        })
        shuffled = rng.permutation(tokens).tolist()
        for fraction in (0.6, 0.8, 1.0):
            prefix = shuffled[:int(len(shuffled) * fraction)]
            records.append({
                **describe(prefix), "target": 1, "group": f"shuffled-{index}",
                "n_tokens": len(prefix), "size_bin": "synthetic-smoke-bin",
            })

    frame = pd.DataFrame(records)
    features = available_features(frame.columns, include_sc=False)
    values = frame[features].to_numpy(dtype=float)
    if not features or not np.isfinite(values).all():
        raise AssertionError("The fixture must produce finite canonical descriptors")
    targets = frame["target"].to_numpy()
    groups = frame["group"].to_numpy()
    out_of_fold = np.full(len(frame), np.nan)
    fold_thresholds = np.full(len(frame), np.nan)
    fold_details = []

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_index, test_index in cv.split(values, targets, groups):
        training = frame.iloc[train_index]
        held_out = frame.iloc[test_index]
        train_groups = set(training["group"])
        test_groups = set(held_out["group"])
        if train_groups & test_groups:
            raise AssertionError("A text group leaked across the validation split")
        if len(np.unique(targets[train_index])) != 2:
            raise AssertionError("Both fixture classes must occur in training")

        # Match the analysis convention: fit reference median/IQR only on the
        # training reference texts, then apply it to both train and held-out rows.
        stats = fit_per_bin_stats(training[training["target"] == 0], features)
        x_train = standardize_per_bin(training, features, stats)
        x_test = standardize_per_bin(held_out, features, stats)
        model = PLSRegression(n_components=2)
        model.fit(x_train, targets[train_index])
        out_of_fold[test_index] = model.predict(x_test).ravel()
        train_targets, train_scores = aggregate_group_scores(
            targets[train_index], model.predict(x_train).ravel(), groups[train_index],
        )
        false_positive, true_positive, thresholds = roc_curve(train_targets, train_scores)
        fold_thresholds[test_index] = thresholds[int(np.argmax(true_positive - false_positive))]
        importance = vip_scores(model)
        if importance.shape != (len(features),) or not np.isfinite(importance).all():
            raise AssertionError("PLS variable importance must be finite")
        fold_details.append({
            "train_groups": len(train_groups), "test_groups": len(test_groups),
            "test_rows": len(test_index),
        })

    if not np.isfinite(out_of_fold).all():
        raise AssertionError("Every row must have a finite held-out score")
    group_targets, group_scores = aggregate_group_scores(targets, out_of_fold, groups)
    _, group_thresholds = aggregate_group_scores(targets, fold_thresholds, groups)
    if len(group_targets) != frame["group"].nunique():
        raise AssertionError("Scoring must count each independent group once")

    return {
        "status": "passed",
        "purpose": "synthetic installation validation; not manuscript results",
        "seed": seed,
        "device": "cpu",
        "tokenizer": tokenizer.metadata(),
        "versions": {name: version(name) for name in (
            "numpy", "scipy", "numba", "torch", "pandas", "scikit-learn", "adjustText",
        )},
        "rows": len(frame),
        "independent_groups": len(group_targets),
        "features": features,
        "folds": fold_details,
        "synthetic_group_auc": float(roc_auc_score(group_targets, group_scores)),
        "synthetic_group_balanced_accuracy": float(balanced_accuracy_score(
            group_targets, group_scores >= group_thresholds,
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        help="Optional JSON report path; must not already exist")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        report = run_validation(args.seed)
    except ModuleNotFoundError as error:
        parser.exit(1, f"Missing dependency {error.name!r}. Install the project "
                    "dependencies first (see README.md).\n")
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

