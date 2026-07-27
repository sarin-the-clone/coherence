#!/usr/bin/env python3
"""
Size-bin-envelope PCA on the SC-analysed natural subcorpus (~1,064 books),
including the canonical feature set: 9 SC + 8 stat/info + 27 token.

This is the SC-inclusive counterpart of pca_allmetrics_size_bins.py: same
combined-figure layout (panel A density + descriptor loadings + size-bin means;
panels B-F per-size-bin natural envelopes with generated prefixes coloured by
ablation mode), but restricted to the books that have SC computed so SC can
enter the PCA.

Reuses data loaders from pca_sc_natural_combined.py and plotting from
pca_allmetrics_size_bins.py.

Outputs (both PC pairs; '_per_bin' suffix when --per-bin-normalization):
  natural_texts/Output/pca/sc_combined/size_bins/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import pca_sc_natural_combined as scn  # noqa: E402
import pca_allmetrics_size_bins as ams  # noqa: E402
from pca_legacy_size_bins import (  # noqa: E402
    assign_bin, fit_per_bin_stats, load_size_bins, size_bin_palette,
    standardize_per_bin,
)

DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "sc_combined" / "size_bins"
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"
DEFAULT_POSTER_DIR = ROOT / "ppt"


def build_natural() -> tuple[pd.DataFrame, list[str]]:
    """Assemble the SC subcorpus with the canonical 35 analysis features."""
    sc_df = scn.load_sc_books()
    batch_df = scn.load_batch_metrics()
    keep_stat = ["book_id"] + [c for c in scn.STAT_INFO_METRICS if c in batch_df.columns]
    batch_df = batch_df[keep_stat].copy()

    tn_df = scn.load_token_jsonl(scn.TOKEN_NAT_JSONL)
    tp_df = scn.load_token_jsonl(scn.TOKEN_PRED_JSONL)
    tn_cols = [c for c in scn.TOKEN_METRICS if c in tn_df.columns]
    tp_cols = [c for c in scn.TOKEN_METRICS if c in tp_df.columns]

    df = (sc_df.merge(batch_df, on="book_id", how="inner")
                .merge(tn_df[["book_id"] + tn_cols], on="book_id", how="inner")
                .merge(tp_df[["book_id"] + tp_cols], on="book_id", how="inner"))

    features = scn.available_features(df.columns)
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["n_tokens"]).reset_index(drop=True)
    return df, features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--poster-dir", type=Path, default=DEFAULT_POSTER_DIR)
    p.add_argument("--per-bin-normalization", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.poster_dir:
        args.poster_dir.mkdir(parents=True, exist_ok=True)

    size_bins = load_size_bins(args.size_stats)
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}

    natural, features = build_natural()
    natural["size_bin"] = [assign_bin(t, size_bins) for t in natural["n_tokens"]]
    natural = natural.dropna(subset=["size_bin"]).reset_index(drop=True)
    print(f"natural SC books: {len(natural)} | features: {len(features)} "
          f"(9 SC + 8 stat + {len(features) - 17} token)")

    generated = scn.load_generated_rows(features)
    for c in features:
        generated[c] = pd.to_numeric(generated[c], errors="coerce")
    generated = generated.replace([np.inf, -np.inf], np.nan).dropna(subset=features).copy()
    generated["size_bin"] = [assign_bin(t, size_bins, lower_edge_tolerance=0.10)
                             for t in generated["n_tokens"]]
    generated = generated.dropna(subset=["size_bin"]).reset_index(drop=True)
    print(f"generated prefixes: {len(generated)} "
          f"({generated['conditioning_mode'].nunique()} modes)")

    if args.per_bin_normalization:
        stats = fit_per_bin_stats(natural, features)
        Zn = standardize_per_bin(natural, features, stats)
        pca = PCA(n_components=4).fit(Zn)
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(Zn)
        generated[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            standardize_per_bin(generated, features, stats))
    else:
        scaler = StandardScaler().fit(natural[features].to_numpy())
        pca = PCA(n_components=4).fit(scaler.transform(natural[features].to_numpy()))
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            scaler.transform(natural[features].to_numpy()))
        generated[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            scaler.transform(generated[features].to_numpy()))
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=features,
                            columns=["PC1", "PC2", "PC3", "PC4"])

    suffix = "_per_bin" if args.per_bin_normalization else ""
    norm_note = "per-size-bin normalized" if args.per_bin_normalization else ""
    n_tok = len(features) - 17
    feat_desc = f"9 SC + 8 stat + {n_tok} token, {len(natural):,} SC books"
    for pcx, pcy, sfx in [("PC1", "PC2", "pc1_pc2"), ("PC3", "PC4", "pc3_pc4")]:
        fig = ams.plot_combined(natural, generated, size_bins, bin_color, evr, loadings,
                                (16.5, 15.5), pcx=pcx, pcy=pcy, norm_note=norm_note,
                                feature_desc=feat_desc)
        name = f"sc_combined_size_bins{suffix}_{sfx}.png"
        fig.savefig(args.output_dir / name, dpi=200)
        print(f"figure  → {args.output_dir / name}")
        if args.poster_dir:
            fig.savefig(args.poster_dir / name, dpi=200)
        plt.close(fig)

    loadings.to_csv(args.output_dir / f"loadings{suffix}.csv")
    import json
    (args.output_dir / f"summary{suffix}.json").write_text(json.dumps({
        "n_natural_books": int(len(natural)),
        "n_generated": int(len(generated)),
        "n_features": len(features),
        "features": features,
        "normalization": "per_bin" if args.per_bin_normalization else "global",
        "explained_variance_ratio": [float(v) for v in evr],
    }, indent=2))


if __name__ == "__main__":
    main()
