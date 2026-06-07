"""
Plot semantic coherence trajectories (novelty and mean sim to past) for one
or more runs, comparing across window scales W1k–W40k.

Usage:
    python analysis/plot_semantic_coherence.py [output.png]

    By default looks for semantic_coherence_trajectories.json in the three
    known run folders and saves to generated_texts_og/Output/semantic_coherence_comparison.png
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS = [
    {
        "path": "generated_texts_og/prompts/"
                "20260518_102401_dune_karikon_s_hidden_secrets_the_aria_chronicles_adventure_extended_conditioning_large_retrieval",
        "label": "Dune (extended conditioning)",
        "color": "#2196F3",   # blue
    },
    {
        "path": "generated_texts_og/prompts/"
                "20260523_063752_last_star_s_rebellion_space_opera_minimal_fixed_conditioning",
        "label": "Last Star (minimal conditioning)",
        "color": "#F44336",   # red
    },
    {
        "path": "generated_texts_og/prompts/"
                "20260525_005409_the_labyrinth_s_last_guardian_awakening_of_the_light_fantasy_markov_rolling_conditioning",
        "label": "Labyrinth (markov rolling)",
        "color": "#FF9800",   # orange
    },
]

WINDOWS = [
    {"key": "W1k",  "chars": 1_000,  "tok": 250,    "label": "W1k\n1k chars / ~250 tok"},
    {"key": "W2k",  "chars": 2_048,  "tok": 512,    "label": "W2k\n2k chars / ~512 tok"},
    {"key": "W8k",  "chars": 8_000,  "tok": 2_000,  "label": "W8k\n8k chars / ~2k tok"},
    {"key": "W40k", "chars": 40_000, "tok": 10_000, "label": "W40k\n40k chars / ~10k tok"},
]


CHARS_PER_GPT4_TOKEN = 4.0

METRICS = [
    {"field": "novelty_trajectory",  "ylabel": "Novelty\n(1 − max sim to past)", "ylim": (0, 1)},
    {"field": "mean_sim_trajectory", "ylabel": "Mean sim\nto past",               "ylim": (0, 1)},
]

SMOOTHING_WINDOW = 10   # rolling average over N windows

OUTPUT_DIR = "generated_texts_og/Output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_traj(run_path):
    p = os.path.join(run_path, "semantic_coherence_trajectories.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def get_series(traj, key_name, field):
    return traj.get(f"{key_name}_{field}")


def char_positions_to_tokens(positions):
    return [p / CHARS_PER_GPT4_TOKEN for p in positions]


def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def smooth_x(x, y, w):
    """Return x and y both smoothed to same length via convolution."""
    ys = smooth(np.array(y, dtype=float), w)
    xs = np.array(x, dtype=float)
    # Trim x to match smoothed y length
    offset = (w - 1) // 2
    xs = xs[offset: offset + len(ys)]
    return xs, ys


def make_xtick_formatter(max_tok):
    """Return a FuncFormatter that labels x in 100k/1M increments."""
    def fmt(x, pos):
        if x == 0:
            return "0"
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        return f"{x/1e3:.0f}k"
    return fmt


# ---------------------------------------------------------------------------
# Build figure: rows = window scales, cols = metrics
# ---------------------------------------------------------------------------

def main(out_path=None):
    n_rows = len(WINDOWS)
    n_cols = len(METRICS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3 * n_rows),
        sharex=False,
        squeeze=False,
    )
    fig.suptitle("Semantic Coherence — Multi-scale Novelty Trajectories", fontsize=13, y=1.01)

    # Track max token count across all runs for consistent x-axis
    max_tokens_per_window = {w["key"]: 0 for w in WINDOWS}

    # First pass: load data and find axis ranges
    run_data = []
    for run in RUNS:
        traj = load_traj(run["path"])
        if traj is None:
            print(f"  [skip] no trajectories.json in {run['path']}")
            run_data.append(None)
            continue
        run_data.append(traj)
        for w in WINDOWS:
            pos = get_series(traj, w["key"], "window_char_positions")
            if pos:
                max_tok = max(char_positions_to_tokens(pos))
                max_tokens_per_window[w["key"]] = max(max_tokens_per_window[w["key"]], max_tok)

    # Second pass: plot
    for row_idx, w in enumerate(WINDOWS):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]

            for run_idx, run in enumerate(RUNS):
                traj = run_data[run_idx]
                if traj is None:
                    continue

                pos = get_series(traj, w["key"], "window_char_positions")
                vals = get_series(traj, w["key"], metric["field"])
                if pos is None or vals is None:
                    continue

                x_tok = char_positions_to_tokens(pos)
                xs, ys = smooth_x(x_tok, vals, SMOOTHING_WINDOW)

                ax.plot(xs, ys, color=run["color"], linewidth=1.2,
                        alpha=0.85, label=run["label"])

            # Check if any data was actually plotted for this window/metric
            has_data = any(
                get_series(run_data[ri], w["key"], metric["field"]) is not None
                for ri in range(len(RUNS)) if run_data[ri] is not None
            )
            if not has_data and col_idx == 0:
                ax.text(0.5, 0.5, "not yet computed\n(re-run with new W1k window)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="gray", style="italic")

            # Y-axis label on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(f"{w['label']}\n\n{metric['ylabel']}", fontsize=8, labelpad=6)
            else:
                ax.set_ylabel(metric["ylabel"], fontsize=8, labelpad=4)

            ax.set_ylim(metric["ylim"])
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
            ax.grid(True, which="minor", alpha=0.1, linewidth=0.4)

            # X-axis: only label bottom row
            max_tok = max_tokens_per_window[w["key"]]
            ax.set_xlim(0, max_tok * 1.02)

            if row_idx == n_rows - 1:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_xtick_formatter(max_tok)))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(100_000))
                ax.set_xlabel("tokens", fontsize=9)
            else:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_xtick_formatter(max_tok)))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(100_000))
                ax.tick_params(axis="x", labelbottom=False)

            # Column title on top row
            if row_idx == 0:
                ax.set_title(metric["ylabel"].replace("\n", " "), fontsize=10)

    # Legend in top-right of first plot
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(handles, labels, fontsize=7, loc="upper right",
                           framealpha=0.8, edgecolor="gray")

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "semantic_coherence_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
