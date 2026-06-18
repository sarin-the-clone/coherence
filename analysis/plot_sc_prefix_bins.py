"""
SC prefix-bin comparison: generated texts vs natural book references.

For each metric × window scale:
  - Natural books as green box plots per bin (loaded from JSON files)
  - Generated runs as box plots per conditioning mode × bin

X-axis = bin (grouped by mode within each bin)
Y-axis = SC scalar value

Usage:
    python analysis/plot_sc_prefix_bins.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INPUT  = "generated_texts_og/Output/sc_prefix_bins.json"
OUTPUT = "generated_texts_og/Output/sc_prefix_bins.png"

# Natural book JSON files (merged at runtime)
NATURAL_JSON_PATHS = [
    "/tmp/natural_books_sc_median.json",
    "/tmp/natural_sc_extra_a.json",
    "/tmp/natural_sc_extra_b.json",
    "/tmp/natural_sc_extra_c.json",
    "/tmp/natural_sc_extra_d.json",
    "/tmp/natural_sc_extra_e.json",
    "/tmp/natural_sc_extra_f.json",
    "/tmp/natural_sc_extra_g.json",
    "/tmp/natural_sc_extra_h.json",
    "/tmp/natural_sc_extra_i.json",
]

WINDOWS = ["W1k", "W2k", "W8k", "W40k"]
WINDOW_LABEL = {"W1k":  "W1k  ~250 tok / ~200 words\n~3–4 paragraphs",
                "W2k":  "W2k  ~512 tok / ~400 words\n~6 paragraphs",
                "W8k":  "W8k  ~2k tok / ~1 600 words\n~a scene",
                "W40k": "W40k  ~10k tok / ~8 000 words\n~2–3 chapters"}

METRICS = [
    ("off_diag_mean",  "Global similarity  (off-diag mean)"),
    ("novelty_mean",   "Novelty  (1 − max sim to past)"),
    ("sim_lag1_mean",  "Local flow  (lag-1 sim)"),
]

BIN_KEYS   = ["0", "1", "2", "3", "4"]
BIN_LABELS = ["100–150k tok", "150–300k tok", "300–500k tok", "500–800k tok", "800k–1M tok"]
BIN_LABELS_SHORT = ["100–150k", "150–300k", "300–500k", "500–800k", "800k–1M"]

MODES_ORDER = [
    "summaries_conditioning",
    "basic_conditioning",
    "extended_conditioning",
    "extended_no_fixed_conditioning",
    "markov_rolling_conditioning",
    "minimal_fixed_conditioning",
]
MODE_COLORS = {
    "summaries_conditioning":      "#1565C0",
    "basic_conditioning":          "#00897B",
    "extended_conditioning":       "#C62828",
    "extended_no_fixed_conditioning": "#00897B",
    "markov_rolling_conditioning": "#E65100",
    "minimal_fixed_conditioning":  "#6A1B9A",
}
MODE_LABELS = {
    "summaries_conditioning":      "summaries",
    "basic_conditioning":          "basic",
    "extended_conditioning":       "extended",
    "extended_no_fixed_conditioning": "extended no fixed",
    "markov_rolling_conditioning": "markov",
    "minimal_fixed_conditioning":  "minimal",
}

NAT_COLOR = "#2E7D32"


def load_natural_by_bin():
    """Merge all natural book JSON files, return {bin_str: [book_dict, ...]}."""
    all_books = {}
    for path in NATURAL_JSON_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                all_books.update(json.load(f))
    by_bin = {}
    for v in all_books.values():
        bk = str(v["bin"])
        if bk in BIN_KEYS:
            by_bin.setdefault(bk, []).append(v)
    return by_bin


def main():
    with open(INPUT) as f:
        data = json.load(f)

    nat_by_bin = load_natural_by_bin()

    # Report how many natural books per bin
    for bk in BIN_KEYS:
        n = len(nat_by_bin.get(bk, []))
        print(f"Bin {bk}: {n} natural books")

    # Group generated runs by conditioning mode
    by_mode = {}
    for run in data.values():
        mode = run.get("conditioning_mode", "?")
        by_mode.setdefault(mode, []).append(run)

    modes_present = [m for m in MODES_ORDER if m in by_mode]
    n_modes = len(modes_present)
    n_bins  = len(BIN_KEYS)

    # Layout: natural box at far left, then generated modes
    # Total slots per bin = n_modes + 1
    n_slots   = n_modes + 1
    total_w   = 0.85
    box_width = total_w / n_slots
    all_offsets = np.linspace(-total_w / 2 + box_width / 2,
                               total_w / 2 - box_width / 2, n_slots)
    nat_offset  = all_offsets[0]
    gen_offsets = all_offsets[1:]

    n_rows = len(WINDOWS)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    fig.suptitle("SC on generated text prefixes vs natural books\n"
                 "(prefix truncated to bin length; box = distribution across runs/books)",
                 fontsize=11, y=1.02)

    for ri, w in enumerate(WINDOWS):
        for ci, (metric, col_label) in enumerate(METRICS):
            ax = axes[ri][ci]
            xs = np.arange(n_bins)

            # ── Natural boxplots (green, far left slot per bin) ──
            nat_median_xs, nat_median_ys = [], []
            for bi, bk in enumerate(BIN_KEYS):
                books = nat_by_bin.get(bk, [])
                vals = [b[w].get(metric) for b in books
                        if w in b and b[w].get(metric) is not None]
                if not vals:
                    continue
                xpos = xs[bi] + nat_offset
                ax.boxplot(vals, positions=[xpos], widths=box_width * 0.85,
                           patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=NAT_COLOR, alpha=0.55),
                           medianprops=dict(color="white", lw=1.8),
                           whiskerprops=dict(color=NAT_COLOR, lw=1.0),
                           capprops=dict(color=NAT_COLOR, lw=1.0),
                           flierprops=dict(marker=".", color=NAT_COLOR,
                                           markersize=3, alpha=0.6),
                           zorder=4)
                nat_median_xs.append(xpos)
                nat_median_ys.append(float(np.median(vals)))

            # connect natural medians across bins
            if len(nat_median_xs) > 1:
                ax.plot(nat_median_xs, nat_median_ys, color=NAT_COLOR,
                        lw=1.5, alpha=0.8, zorder=5, linestyle="--")

            # ── Generated boxplots (one slot per mode) ──
            for mi, mode in enumerate(modes_present):
                color = MODE_COLORS[mode]
                runs  = by_mode[mode]
                median_xs, median_ys = [], []
                for bi, bk in enumerate(BIN_KEYS):
                    vals = [r.get("bins", {}).get(bk, {}).get(w, {}).get(metric)
                            for r in runs]
                    vals = [v for v in vals if v is not None]
                    if not vals:
                        continue
                    xpos = xs[bi] + gen_offsets[mi]
                    ax.boxplot(vals, positions=[xpos], widths=box_width * 0.85,
                               patch_artist=True, manage_ticks=False,
                               boxprops=dict(facecolor=color, alpha=0.6),
                               medianprops=dict(color="white", lw=1.5),
                               whiskerprops=dict(color=color, lw=1.0),
                               capprops=dict(color=color, lw=1.0),
                               flierprops=dict(marker=".", color=color,
                                               markersize=3, alpha=0.5),
                               zorder=3)
                    median_xs.append(xpos)
                    median_ys.append(float(np.median(vals)))
                if len(median_xs) > 1:
                    ax.plot(median_xs, median_ys, color=color, lw=1.2,
                            alpha=0.8, zorder=4, linestyle="--")

            ax.set_xticks(xs)
            ax.set_xticklabels(BIN_LABELS if ri == n_rows - 1 else
                               BIN_LABELS_SHORT, fontsize=7)
            ax.margins(y=0.15)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 5]))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.grid(True, axis="y", which="major", alpha=0.2, lw=0.6)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(WINDOW_LABEL[w], fontsize=8)
            if ri == 0:
                ax.set_title(col_label, fontsize=10)

    # Legend
    nat_count = {bk: len(nat_by_bin.get(bk, [])) for bk in BIN_KEYS}
    min_nat = min(nat_count.values()) if nat_count else 0
    max_nat = max(nat_count.values()) if nat_count else 0
    nat_label = (f"natural ({min_nat}–{max_nat} books/bin)"
                 if min_nat != max_nat else f"natural ({min_nat} books/bin)")
    handles = [Patch(facecolor=NAT_COLOR, alpha=0.6, label=nat_label)]
    for mode in modes_present:
        handles.append(Patch(facecolor=MODE_COLORS[mode], alpha=0.7,
                             label=MODE_LABELS[mode]))
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.02), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
