"""
Sanity-check plot: natural bin staircase vs 3 picked generated books
(scalar values shown as horizontal lines).

Rows = W1k, W2k, W8k, W40k
Cols = off_diag_mean | novelty_mean | sim_lag1_mean

Usage:
    python analysis/plot_sc_batch_check.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Natural staircase (median book per bin)
# ---------------------------------------------------------------------------

BIN_EDGES = [100_000, 150_000, 300_000, 500_000, 800_000, 1_000_000]

NATURAL = {
    # 3-book mean per bin (stride=1.0)
    0: {"label": "natural bin 0 (3 books)",
        "W1k":  {"off_diag_mean": 0.261, "novelty_mean": 0.441, "sim_lag1_mean": 0.304},
        "W2k":  {"off_diag_mean": 0.398, "novelty_mean": 0.404, "sim_lag1_mean": 0.451},
        "W8k":  {"off_diag_mean": 0.680, "novelty_mean": 0.225, "sim_lag1_mean": 0.727},
        "W40k": {"off_diag_mean": 0.887, "novelty_mean": 0.085, "sim_lag1_mean": 0.899},
    },
    1: {"label": "natural bin 1 (3 books)",
        "W1k":  {"off_diag_mean": 0.290, "novelty_mean": 0.413, "sim_lag1_mean": 0.328},
        "W2k":  {"off_diag_mean": 0.433, "novelty_mean": 0.376, "sim_lag1_mean": 0.478},
        "W8k":  {"off_diag_mean": 0.721, "novelty_mean": 0.199, "sim_lag1_mean": 0.757},
        "W40k": {"off_diag_mean": 0.892, "novelty_mean": 0.076, "sim_lag1_mean": 0.915},
    },
    2: {"label": "natural bin 2 (3 books)",
        "W1k":  {"off_diag_mean": 0.252, "novelty_mean": 0.397, "sim_lag1_mean": 0.300},
        "W2k":  {"off_diag_mean": 0.382, "novelty_mean": 0.378, "sim_lag1_mean": 0.439},
        "W8k":  {"off_diag_mean": 0.673, "novelty_mean": 0.205, "sim_lag1_mean": 0.730},
        "W40k": {"off_diag_mean": 0.862, "novelty_mean": 0.081, "sim_lag1_mean": 0.907},
    },
    3: {"label": "natural bin 3 (3 books)",
        "W1k":  {"off_diag_mean": 0.280, "novelty_mean": 0.373, "sim_lag1_mean": 0.345},
        "W2k":  {"off_diag_mean": 0.409, "novelty_mean": 0.350, "sim_lag1_mean": 0.491},
        "W8k":  {"off_diag_mean": 0.679, "novelty_mean": 0.191, "sim_lag1_mean": 0.755},
        "W40k": {"off_diag_mean": 0.844, "novelty_mean": 0.081, "sim_lag1_mean": 0.903},
    },
    4: {"label": "natural bin 4 (3 books)",
        "W1k":  {"off_diag_mean": 0.302, "novelty_mean": 0.333, "sim_lag1_mean": 0.355},
        "W2k":  {"off_diag_mean": 0.443, "novelty_mean": 0.321, "sim_lag1_mean": 0.506},
        "W8k":  {"off_diag_mean": 0.710, "novelty_mean": 0.170, "sim_lag1_mean": 0.778},
        "W40k": {"off_diag_mean": 0.871, "novelty_mean": 0.063, "sim_lag1_mean": 0.920},
    },
}

# ---------------------------------------------------------------------------
# Generated books (3 picks, scalars only)
# ---------------------------------------------------------------------------

GENERATED_PICKS = [
    {"label": "Moonlit Conquest\n(summaries)",  "color": "#2196F3",
     "run_id": "20260204_155340_the_moonlit_conquest_shadows_of_the_city_science_summaries_conditioning_high_temperature"},
    {"label": "Forgotten God\n(extended)",       "color": "#F44336",
     "run_id": "20260517_184348_the_forgotten_god_s_reckoning_fantasy_extended_conditioning"},
    {"label": "Heart of Darkness\n(markov)",     "color": "#FF9800",
     "run_id": "20260524_090540_heart_of_darkness_space_opera_crusade_adventure_markov_rolling_conditioning"},
]

WINDOWS = ["W1k", "W2k", "W8k", "W40k"]
WINDOW_LABEL = {"W1k": "W1k  ~250 tok", "W2k": "W2k  ~512 tok",
                "W8k": "W8k  ~2k tok",  "W40k": "W40k  ~10k tok"}

METRICS = [
    ("off_diag_mean",  "Global similarity  (off-diag mean)"),
    ("novelty_mean",   "Novelty  (1 − max sim to past)"),
    ("sim_lag1_mean",  "Local flow  (lag-1 sim)"),
]

NAT_COLOR = "#2E7D32"
NAT_LW    = 2.5
NAT_ALPHA = 0.85
OUTPUT    = "generated_texts_og/Output/sc_batch_check.png"


def draw_staircase(ax, window, metric):
    for b_idx in sorted(NATURAL.keys()):
        val = NATURAL[b_idx][window].get(metric, float("nan"))
        if np.isnan(val):
            continue
        x_lo, x_hi = BIN_EDGES[b_idx], BIN_EDGES[b_idx + 1]
        ax.hlines(val, x_lo, x_hi, colors=NAT_COLOR, lw=NAT_LW, alpha=NAT_ALPHA, zorder=2)
        if b_idx + 1 in NATURAL:
            val_next = NATURAL[b_idx + 1][window].get(metric, float("nan"))
            if not np.isnan(val_next):
                ax.vlines(x_hi, min(val, val_next), max(val, val_next),
                          colors=NAT_COLOR, lw=NAT_LW * 0.6,
                          alpha=NAT_ALPHA * 0.6, linestyle="--", zorder=2)


def main():
    with open("generated_texts_og/Output/sc_batch_results.json") as f:
        batch = json.load(f)

    n_rows, n_cols = len(WINDOWS), len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 2.6 * n_rows), squeeze=False)
    fig.suptitle("SC sanity check — 3 generated books vs natural bin means",
                 fontsize=12, y=1.01)

    for ri, w in enumerate(WINDOWS):
        for ci, (metric, col_label) in enumerate(METRICS):
            ax = axes[ri][ci]
            draw_staircase(ax, w, metric)

            for pick in GENERATED_PICKS:
                run = batch.get(pick["run_id"], {})
                val = run.get(w, {}).get(metric, float("nan"))
                if not np.isnan(val):
                    ax.axhline(val, color=pick["color"], lw=1.5, linestyle="--",
                               alpha=0.85, zorder=3,
                               label=pick["label"] if ri == 0 and ci == 0 else "")

            ax.margins(y=0.15)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 5]))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.grid(True, which="major", alpha=0.2, lw=0.6)
            ax.grid(True, which="minor", alpha=0.08, lw=0.4)
            ax.set_xlim(0, 1_000_000)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(200_000))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: "1M" if x >= 1e6 else f"{x/1e3:.0f}k"))
            ax.tick_params(labelsize=7)
            if ri == n_rows - 1:
                ax.set_xlabel("tokens", fontsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if ci == 0:
                ax.set_ylabel(WINDOW_LABEL[w], fontsize=8)
            if ri == 0:
                ax.set_title(col_label, fontsize=10)

    gen_handles, gen_labels = axes[0][0].get_legend_handles_labels()
    nat_proxy = Line2D([0], [0], color=NAT_COLOR, lw=NAT_LW,
                       label="natural (median book/bin)")
    fig.legend(handles=gen_handles + [nat_proxy],
               labels=gen_labels + [nat_proxy.get_label()],
               loc="upper right", fontsize=7,
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
