"""
Plot 2: Semantic coherence trajectories as generated text grows, with natural
book trajectories overlaid for reference.

Rows = window scales (W2k, W8k, W40k)
Cols = novelty trajectory | mean-sim-to-past trajectory

Generated runs: full 1M-token trajectory lines.
Natural books: short trajectory lines drawn at their actual character positions,
colour-coded by size bin.

Usage:
    python analysis/plot_sc_trajectories_with_nat.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENERATED_RUNS = [
    {"label": "Dune (ext-cond)",    "color": "#2196F3",
     "path": "generated_texts_og/prompts/20260518_102401_dune_karikon_s_hidden_secrets_the_aria_chronicles_adventure_extended_conditioning_large_retrieval"},
    {"label": "Last Star (min-cond)","color": "#F44336",
     "path": "generated_texts_og/prompts/20260523_063752_last_star_s_rebellion_space_opera_minimal_fixed_conditioning"},
    {"label": "Labyrinth (mrk-roll)","color": "#FF9800",
     "path": "generated_texts_og/prompts/20260525_005409_the_labyrinth_s_last_guardian_awakening_of_the_light_fantasy_markov_rolling_conditioning"},
]

# natural books hard-coded; bins 2-4 loaded from /tmp/natural_books_sc.json
NATURAL_HARD = [
    {"label": "Nobody's Child (110k)", "bin": 0, "n_tokens": 110_360, "color": "#388E3C",
     "W2k":  {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
     "W8k":  {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
     "W40k": {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
    },
    {"label": "Fathom (243k)",         "bin": 1, "n_tokens": 242_727, "color": "#66BB6A",
     "W2k":  {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
     "W8k":  {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
     "W40k": {"novelty_trajectory": None, "mean_sim_trajectory": None, "window_char_positions": None},
    },
]
# load trajectory arrays for hard-coded books from their trajectory JSONs
# (we recomputed them earlier; they're in the run output dir or /tmp)
# We'll pull them from /tmp/natural_books_sc.json + hard-coded for the two we have
HARD_TRAJ_PATH = None  # populated at runtime from /tmp

NAT_BIN_COLORS = {0: "#1B5E20", 1: "#388E3C", 2: "#66BB6A", 3: "#A5D6A7", 4: "#C8E6C9"}

WINDOWS = ["W2k", "W8k", "W40k"]
WINDOW_CHARS_APPROX = {"W2k": 2048, "W8k": 8000, "W40k": 40000}
CHARS_PER_TOKEN = 4.0
SMOOTHING = 10

METRICS = [
    ("novelty_trajectory",  "Novelty (1 − max sim to past)"),
    ("mean_sim_trajectory", "Mean sim to past"),
]

OUTPUT = "generated_texts_og/Output/sc_trajectories_with_nat.png"


def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(np.array(arr, dtype=float), np.ones(w)/w, mode="valid")


def trim_x(x, y, w):
    ys = smooth(y, w)
    offset = (w - 1) // 2
    xs = np.array(x, dtype=float)[offset: offset + len(ys)]
    return xs / CHARS_PER_TOKEN, ys   # convert chars → tokens


def load_generated():
    runs = []
    for run in GENERATED_RUNS:
        with open(os.path.join(run["path"], "semantic_coherence_trajectories.json")) as f:
            t = json.load(f)
        entry = dict(run)
        for w in WINDOWS:
            pos  = t.get(f"{w}_window_char_positions", [])
            nov  = t.get(f"{w}_novelty_trajectory", [])
            msp  = t.get(f"{w}_mean_sim_trajectory", [])
            entry[w] = {"pos": pos, "novelty_trajectory": nov, "mean_sim_trajectory": msp}
        runs.append(entry)
    return runs


def load_natural():
    # Bins 2-4 from computed JSON
    extra_path = "/tmp/natural_books_sc.json"
    extra = []
    if os.path.exists(extra_path):
        with open(extra_path) as f:
            data = json.load(f)
        for bid, d in data.items():
            entry = {
                "label": f"{d['title'][:18]} ({d['n_tokens']//1000}k)",
                "bin":   d["bin"],
                "n_tokens": d["n_tokens"],
                "color": NAT_BIN_COLORS.get(d["bin"], "#A5D6A7"),
            }
            for w in WINDOWS:
                if w in d and "novelty_trajectory" in d[w]:
                    entry[w] = {
                        "pos": d[w]["window_char_positions"],
                        "novelty_trajectory":  d[w]["novelty_trajectory"],
                        "mean_sim_trajectory": d[w]["mean_sim_trajectory"],
                    }
            extra.append(entry)

    # Hard-coded bins 0-1: trajectories stored in the run output or must be
    # re-read from the SemanticCoherence result. Since we don't have a saved
    # trajectory JSON for natural books yet, we skip trajectory for them and
    # instead just draw a vertical span using their off_diag_mean as reference.
    # (Once the corpus script saves per-book JSONs, these can be wired in.)
    hard = [
        {"label": "Nobody's Child (110k)", "bin": 0, "n_tokens": 110_360,
         "color": NAT_BIN_COLORS[0],
         "W2k_odm": 0.414, "W8k_odm": 0.707, "W40k_odm": 0.909},
        {"label": "Fathom (243k)",         "bin": 1, "n_tokens": 242_727,
         "color": NAT_BIN_COLORS[1],
         "W2k_odm": 0.423, "W8k_odm": 0.729, "W40k_odm": 0.909},
    ]
    return sorted(extra, key=lambda x: x["n_tokens"]), hard


def main():
    gen_runs  = load_generated()
    nat_extra, nat_hard = load_natural()

    # Also load trajectories for nat_extra books
    # (they have pos/novelty/mean_sim in entry[w])

    n_rows = len(WINDOWS)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3 * n_rows),
                             squeeze=False)
    fig.suptitle("Semantic Coherence Trajectories — Generated vs Natural", fontsize=13, y=1.01)

    for ri, w in enumerate(WINDOWS):
        for ci, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[ri][ci]

            # ── generated runs ──
            for run in gen_runs:
                wdata = run.get(w, {})
                pos  = wdata.get("pos", [])
                vals = wdata.get(metric_key, [])
                if not pos or not vals:
                    continue
                xs, ys = trim_x(pos, vals, SMOOTHING)
                ax.plot(xs, ys, color=run["color"], lw=1.3, alpha=0.85,
                        label=run["label"] if ri == 0 and ci == 0 else "")

            # ── natural book trajectories (bins 2-4) ──
            for nb in nat_extra:
                wdata = nb.get(w, {})
                pos  = wdata.get("pos", [])
                vals = wdata.get(metric_key, [])
                if not pos or not vals:
                    continue
                xs, ys = trim_x(pos, vals, max(1, SMOOTHING // 3))
                ax.plot(xs, ys, color=nb["color"], lw=1.5, alpha=0.7,
                        linestyle="--",
                        label=nb["label"] if ri == 0 and ci == 0 else "")

            # ── natural off_diag_mean as horizontal reference lines ──
            # (for bins 0-1 where we have scalars but not trajectories here)
            # off_diag_mean maps to mean_sim side; for novelty use 1-odm approx
            for nb in nat_hard:
                odm = nb.get(f"{w}_odm", float("nan"))
                if np.isnan(odm):
                    continue
                ref = odm if metric_key == "mean_sim_trajectory" else (1 - odm)
                ax.axhline(ref, color=nb["color"], lw=1.2, linestyle=":",
                           alpha=0.8,
                           label=f"{nb['label']} ref" if ri == 0 and ci == 0 else "")

            # axis
            if ri == 0:
                ax.set_title(metric_label, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{w}\n{['~250','~512','~2k','~10k'][['W1k','W2k','W8k','W40k'].index(w)]} tok",
                              fontsize=8)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.grid(True, alpha=0.2)

            if ri == n_rows - 1:
                ax.set_xlabel("tokens", fontsize=8)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(100_000))
            else:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(100_000))
                ax.tick_params(axis="x", labelbottom=False)

    # legend
    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[0][1].get_legend_handles_labels()
    all_h = h0 + [x for x, lb in zip(h1, l1) if lb not in l0]
    all_l = l0 + [lb for lb in l1 if lb not in l0]
    if all_h:
        fig.legend(all_h, all_l, loc="upper right", fontsize=7,
                   bbox_to_anchor=(1.18, 1.0), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
