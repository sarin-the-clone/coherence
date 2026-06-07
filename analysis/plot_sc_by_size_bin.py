"""
Plot 1: Mean semantic coherence metrics per size bin × window scale.

One natural book per bin (stand-in for the bin mean), plus the 3 generated
runs shown as a separate "generated" group on the right.

Usage:
    python analysis/plot_sc_by_size_bin.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Natural books — one per bin (computed via SemanticCoherence retrospective)
NATURAL_BOOKS = [
    {"label": "110k\nNobody's Child",  "bin": 0, "n_tokens": 110_360,
     "W1k":  {"off_diag_mean": 0.281, "novelty_mean": 0.389, "novelty_std": 0.116, "sim_lag1_mean": 0.351, "msp_final": 0.344},
     "W2k":  {"off_diag_mean": 0.414, "novelty_mean": 0.275, "novelty_std": 0.067, "sim_lag1_mean": 0.716, "msp_final": 0.366},
     "W8k":  {"off_diag_mean": 0.707, "novelty_mean": 0.203, "novelty_std": 0.035, "sim_lag1_mean": 0.772, "msp_final": 0.710},
     "W40k": {"off_diag_mean": 0.909, "novelty_mean": 0.061, "novelty_std": 0.010, "sim_lag1_mean": 0.935, "msp_final": 0.900},
    },
    {"label": "243k\nFathom",          "bin": 1, "n_tokens": 242_727,
     "W1k":  {"off_diag_mean": 0.281, "novelty_mean": 0.384, "novelty_std": 0.120, "sim_lag1_mean": 0.333, "msp_final": 0.338},
     "W2k":  {"off_diag_mean": 0.423, "novelty_mean": 0.269, "novelty_std": 0.066, "sim_lag1_mean": 0.718, "msp_final": 0.374},
     "W8k":  {"off_diag_mean": 0.729, "novelty_mean": 0.182, "novelty_std": 0.029, "sim_lag1_mean": 0.785, "msp_final": 0.693},
     "W40k": {"off_diag_mean": 0.909, "novelty_mean": 0.060, "novelty_std": 0.009, "sim_lag1_mean": 0.936, "msp_final": 0.904},
    },
    # bins 2-4 loaded from /tmp/natural_books_sc.json at runtime
]

GENERATED_RUNS = [
    {"label": "Dune\n(ext-cond)",    "color": "#2196F3", "n_tokens": 1_086_976,
     "W2k":  {"off_diag_mean": 0.572, "novelty_mean": 0.232, "novelty_std": 0.060, "msp_final": 0.541},
     "W8k":  {"off_diag_mean": 0.721, "novelty_mean": 0.180, "novelty_std": 0.034, "msp_final": 0.674},
     "W40k": {"off_diag_mean": 0.889, "novelty_mean": 0.079, "novelty_std": 0.020, "msp_final": 0.882},
    },
    {"label": "Last Star\n(min-cond)","color": "#F44336", "n_tokens": 1_092_352,
     "W2k":  {"off_diag_mean": 0.512, "novelty_mean": 0.275, "novelty_std": 0.057, "msp_final": 0.542},
     "W8k":  {"off_diag_mean": 0.722, "novelty_mean": 0.190, "novelty_std": 0.023, "msp_final": 0.649},
     "W40k": {"off_diag_mean": 0.923, "novelty_mean": 0.057, "novelty_std": 0.006, "msp_final": 0.905},
    },
    {"label": "Labyrinth\n(mrk-roll)","color": "#FF9800", "n_tokens": 1_086_464,
     "W2k":  {"off_diag_mean": 0.400, "novelty_mean": 0.266, "novelty_std": 0.082, "msp_final": 0.471},
     "W8k":  {"off_diag_mean": 0.548, "novelty_mean": 0.205, "novelty_std": 0.054, "msp_final": 0.625},
     "W40k": {"off_diag_mean": 0.699, "novelty_mean": 0.118, "novelty_std": 0.046, "msp_final": 0.772},
    },
]

WINDOWS = ["W1k", "W2k", "W8k", "W40k"]
WINDOW_TOKENS = {"W1k": "~250 tok", "W2k": "~512 tok", "W8k": "~2k tok", "W40k": "~10k tok"}

METRICS = [
    ("off_diag_mean", "Global similarity\n(off-diag mean sim, ↓ = more diverse)"),
    ("novelty_std",   "Novelty variability\n(std of novelty trajectory)"),
    ("msp_final",     "Convergence\n(mean-sim-to-past at end, ↑ = stagnated)"),
]

NAT_COLOR  = "#4CAF50"
NAT_MARKER = "o"
GEN_MARKER = "^"

OUTPUT = "generated_texts_og/Output/sc_by_size_bin.png"


def load_extra_books():
    path = "/tmp/natural_books_sc.json"
    if not os.path.exists(path):
        print(f"[warn] {path} not found — only bins 0-1 will be shown")
        return []
    with open(path) as f:
        data = json.load(f)
    extra = []
    for bid, d in data.items():
        entry = {
            "label": f"{d['n_tokens']//1000}k\n{d['title'][:14]}",
            "bin":   d["bin"],
            "n_tokens": d["n_tokens"],
        }
        for w in WINDOWS:
            if w in d:
                entry[w] = {
                    "off_diag_mean": d[w].get("off_diag_mean", float("nan")),
                    "novelty_mean":  d[w].get("novelty_mean",  float("nan")),
                    "novelty_std":   d[w].get("novelty_std",   float("nan")),
                    "sim_lag1_mean": d[w].get("sim_lag1_mean", float("nan")),
                    "msp_final":     d[w].get("mean_sim_to_past_final", float("nan")),
                }
        extra.append(entry)
    return sorted(extra, key=lambda x: x["n_tokens"])


def main():
    extra = load_extra_books()
    all_nat = sorted(NATURAL_BOOKS + extra, key=lambda x: x["n_tokens"])

    n_metrics = len(METRICS)
    n_windows = len(WINDOWS)

    fig, axes = plt.subplots(n_metrics, n_windows,
                             figsize=(3.5 * n_windows, 3.2 * n_metrics),
                             squeeze=False)
    fig.suptitle("Semantic Coherence by Text Size and Window Scale", fontsize=13, y=1.01)

    for mi, (metric_key, metric_label) in enumerate(METRICS):
        for wi, w in enumerate(WINDOWS):
            ax = axes[mi][wi]

            # natural books as dots connected by line
            nat_x = [b["n_tokens"] / 1000 for b in all_nat if w in b]
            nat_y = [b[w].get(metric_key, float("nan")) for b in all_nat if w in b]
            nat_labels = [b["label"] for b in all_nat if w in b]

            ax.plot(nat_x, nat_y, color=NAT_COLOR, linewidth=1.2, alpha=0.6, zorder=1)
            ax.scatter(nat_x, nat_y, color=NAT_COLOR, s=55, zorder=3,
                       marker=NAT_MARKER, label="natural (1 book/bin)" if wi == 0 and mi == 0 else "")

            # generated runs as triangles at their token count
            for run in GENERATED_RUNS:
                v = run.get(w, {}).get(metric_key, float("nan"))
                if not np.isnan(v):
                    ax.scatter(run["n_tokens"] / 1000, v,
                               color=run["color"], s=80, zorder=4,
                               marker=GEN_MARKER,
                               label=run["label"] if wi == 0 and mi == 0 else "")

            # axis formatting
            if mi == 0:
                ax.set_title(f"{w}\n{WINDOW_TOKENS[w]}", fontsize=9)
            if wi == 0:
                ax.set_ylabel(metric_label, fontsize=8)

            ax.set_xlabel("tokens (k)" if mi == n_metrics - 1 else "", fontsize=8)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{x/1000:.0f}M" if x >= 1000 else f"{x:.0f}k"))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

            # skip W1k for generated (not computed)
            if w == "W1k":
                ax.text(0.5, 0.5, "generated runs\nnot computed at W1k",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="gray", style="italic")

    # single legend
    handles, labels = axes[0][1].get_legend_handles_labels()
    h0, l0 = axes[0][0].get_legend_handles_labels()
    all_h = h0 + handles
    all_l = l0 + labels
    if all_h:
        fig.legend(all_h, all_l, loc="upper right", fontsize=7,
                   bbox_to_anchor=(1.01, 1.0), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
