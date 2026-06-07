"""
Semantic coherence staircase plot for off_diag_mean and sim_lag1_mean.

Rows = window scales (W1k, W2k, W8k, W40k)
Cols = off_diag_mean | sim_lag1_mean

Natural books: staircase (one median-representative book per size bin).
Generated runs:
  - off_diag_mean: cumulative trajectory derived from stored mean_sim_trajectory
  - sim_lag1_mean: not available from online trajectory data

Usage:
    python analysis/plot_sc_odm_lag1.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BIN_EDGES = [100_000, 150_000, 300_000, 500_000, 800_000, 1_000_000]

NATURAL_HARD = {
    # 3-book mean per bin (stride=1.0)
    0: {"n_tokens": 121_140, "label": "natural bin 0 (3 books)",
        "W1k":  {"off_diag_mean": 0.261, "sim_lag1_mean": 0.304},
        "W2k":  {"off_diag_mean": 0.398, "sim_lag1_mean": 0.451},
        "W8k":  {"off_diag_mean": 0.680, "sim_lag1_mean": 0.727},
        "W40k": {"off_diag_mean": 0.887, "sim_lag1_mean": 0.899},
    },
    1: {"n_tokens": 193_795, "label": "natural bin 1 (3 books)",
        "W1k":  {"off_diag_mean": 0.290, "sim_lag1_mean": 0.328},
        "W2k":  {"off_diag_mean": 0.433, "sim_lag1_mean": 0.478},
        "W8k":  {"off_diag_mean": 0.721, "sim_lag1_mean": 0.757},
        "W40k": {"off_diag_mean": 0.892, "sim_lag1_mean": 0.915},
    },
    2: {"n_tokens": 353_917, "label": "natural bin 2 (3 books)",
        "W1k":  {"off_diag_mean": 0.252, "sim_lag1_mean": 0.300},
        "W2k":  {"off_diag_mean": 0.382, "sim_lag1_mean": 0.439},
        "W8k":  {"off_diag_mean": 0.673, "sim_lag1_mean": 0.730},
        "W40k": {"off_diag_mean": 0.862, "sim_lag1_mean": 0.907},
    },
    3: {"n_tokens": 593_869, "label": "natural bin 3 (3 books)",
        "W1k":  {"off_diag_mean": 0.280, "sim_lag1_mean": 0.345},
        "W2k":  {"off_diag_mean": 0.409, "sim_lag1_mean": 0.491},
        "W8k":  {"off_diag_mean": 0.679, "sim_lag1_mean": 0.755},
        "W40k": {"off_diag_mean": 0.844, "sim_lag1_mean": 0.903},
    },
    4: {"n_tokens": 1_072_000, "label": "natural bin 4 (3 books)",
        "W1k":  {"off_diag_mean": 0.302, "sim_lag1_mean": 0.355},
        "W2k":  {"off_diag_mean": 0.443, "sim_lag1_mean": 0.506},
        "W8k":  {"off_diag_mean": 0.710, "sim_lag1_mean": 0.778},
        "W40k": {"off_diag_mean": 0.871, "sim_lag1_mean": 0.920},
    },
}

GENERATED_RUNS = [
    # summaries_conditioning — blue
    {"label": "Awakening Heart (sum)", "color": "#1565C0",
     "path": "generated_texts_og/prompts/20260102_030044_awakening_the_heart_aria_mira_s_quest_to_restore_zyrathos_historical_fiction_summaries_conditioning_high_temperature"},
    {"label": "Shadows Rising (sum)",  "color": "#1976D2",
     "path": "generated_texts_og/prompts/20260105_130607_shadows_rising_redemption_s_dawn_neo_tokyo_rising_thriller_summaries_conditioning_high_temperature"},
    {"label": "Lady Byron (sum)",      "color": "#42A5F5",
     "path": "generated_texts_og/prompts/20260206_113034_lady_byron_s_gothic_library_ada_s_enlightened_obsession_biography_summaries_conditioning_high_temperature"},
    # basic_conditioning — teal
    {"label": "Love Whirlwind (bas)",  "color": "#00695C",
     "path": "generated_texts_og/prompts/20260521_005834_love_s_whirlwind_a_tale_of_resilience_and_redemption_romance_basic_conditioning"},
    {"label": "Shadows in Neon (bas)", "color": "#00897B",
     "path": "generated_texts_og/prompts/20260527_072914_shadows_in_neon_rebellion_s_dawn_thriller_basic_conditioning"},
    {"label": "Keeper Whispers (bas)", "color": "#4DB6AC",
     "path": "generated_texts_og/prompts/20260529_204415_the_keeper_of_whispers_and_shadows_the_lighthouse_enigma_fantasy_basic_conditioning"},
    # extended_conditioning — red
    {"label": "Dune (ext)",            "color": "#B71C1C",
     "path": "generated_texts_og/prompts/20260518_102401_dune_karikon_s_hidden_secrets_the_aria_chronicles_adventure_extended_conditioning_large_retrieval"},
    {"label": "Artifact Shadow (ext)", "color": "#D32F2F",
     "path": "generated_texts_og/prompts/20260515_083256_the_artifact_s_shadow_a_tale_from_neo_tokyo_s_crypts_thriller_extended_conditioning_small_retrieval"},
    {"label": "Forgotten God (ext)",   "color": "#EF9A9A",
     "path": "generated_texts_og/prompts/20260517_184348_the_forgotten_god_s_reckoning_fantasy_extended_conditioning_small_retrieval"},
    # markov_rolling_conditioning — orange
    {"label": "Labyrinth (mrk)",       "color": "#E65100",
     "path": "generated_texts_og/prompts/20260525_005409_the_labyrinth_s_last_guardian_awakening_of_the_light_fantasy_markov_rolling_conditioning"},
    {"label": "Stellar Exodus (mrk)",  "color": "#FF6D00",
     "path": "generated_texts_og/prompts/20260521_144550_stellar_exodus_the_final_frontier_space_opera_markov_rolling_conditioning"},
    # minimal_fixed_conditioning — purple
    {"label": "Last Star (min)",       "color": "#6A1B9A",
     "path": "generated_texts_og/prompts/20260523_063752_last_star_s_rebellion_space_opera_minimal_fixed_conditioning"},
    {"label": "Preoom Repair (min)",   "color": "#AB47BC",
     "path": "generated_texts_og/prompts/20260522_043453_preoom_repair_the_inventor_s_dilemma_a_guide_to_robotics_and_the_ethics_of_artificial_life_science_minimal_fixed_conditioning"},
]

WINDOWS = ["W1k", "W2k", "W8k", "W40k"]
WINDOW_LABEL = {"W1k": "W1k  ~250 tok", "W2k": "W2k  ~512 tok",
                "W8k": "W8k  ~2k tok",  "W40k": "W40k  ~10k tok"}

CHARS_PER_TOKEN = 4.0
SMOOTHING       = 20
NAT_COLOR       = "#2E7D32"
NAT_LW          = 2.5
NAT_ALPHA       = 0.85

OUTPUT = "generated_texts_og/Output/sc_odm_lag1.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cumulative_odm(msp_traj):
    """Running off_diag_mean from mean_sim_to_past trajectory.

    msp_traj[0] = NaN (window 0 has no past).
    msp_traj[k] = mean(sim[k, 0..k-1]) for k >= 1.

    off_diag_mean(M windows) = 2 * sum_{k=1}^{M-1}(k * msp[k]) / (M*(M-1))

    Returns array of length n-1: result[j] = odm after seeing windows 0..j+1
    (i.e., after M = j+2 windows), aligned to pos[1:].
    """
    msp = np.asarray(msp_traj, dtype=float)
    n = len(msp)
    if n < 2:
        return np.array([])
    k = np.arange(1, n)          # k = 1..n-1
    cum = np.cumsum(k * msp[1:]) # Σ k*msp[k], skipping NaN at index 0
    M = k + 1                    # M = 2..n windows seen
    return 2.0 * cum / (M * (M - 1))  # length n-1


def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(np.array(arr, dtype=float), np.ones(w) / w, mode="valid")


def smoothed_xy(pos_tok, vals, w):
    effective_w = w if len(vals) >= w else 1
    ys = smooth(vals, effective_w)
    offset = (effective_w - 1) // 2
    xs = pos_tok[offset: offset + len(ys)]
    return xs, ys


def load_natural_bins():
    bins = dict(NATURAL_HARD)
    path = "/tmp/natural_books_sc_median.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for bid, d in data.items():
            b = d["bin"]
            entry = {"n_tokens": d["n_tokens"], "label": d["title"][:20]}
            for w in WINDOWS:
                if w in d:
                    entry[w] = {
                        "off_diag_mean": d[w].get("off_diag_mean", float("nan")),
                        "sim_lag1_mean": d[w].get("sim_lag1_mean", float("nan")),
                    }
            bins[b] = entry
    return bins


def load_generated():
    runs = []
    for run in GENERATED_RUNS:
        with open(os.path.join(run["path"], "semantic_coherence_trajectories.json")) as f:
            t = json.load(f)
        entry = dict(run)
        for w in WINDOWS:
            pos = np.array(t.get(f"{w}_window_char_positions", []), dtype=float)
            msp = np.array(t.get(f"{w}_mean_sim_trajectory",   []), dtype=float)
            if len(pos) and len(msp):
                odm  = cumulative_odm(msp)
                lag1 = np.array(t.get(f"{w}_sim_lag1_trajectory", []), dtype=float)
                # odm has length n-1; align to pos[1:] (skip window-0 position)
                entry[w] = {
                    "pos_tok":         pos[1:] / CHARS_PER_TOKEN,
                    "odm_trajectory":  odm,
                    "lag1_trajectory": lag1[1:] if len(lag1) > 1 else np.array([]),
                }
        runs.append(entry)
    return runs


def draw_staircase(ax, nat_bins, window, metric_key):
    for b_idx in sorted(nat_bins.keys()):
        bdata = nat_bins[b_idx]
        if window not in bdata:
            continue
        val = bdata[window].get(metric_key, float("nan"))
        if np.isnan(val):
            continue
        x_lo = BIN_EDGES[b_idx]
        x_hi = BIN_EDGES[b_idx + 1]
        ax.hlines(val, x_lo, x_hi, colors=NAT_COLOR, linewidth=NAT_LW,
                  alpha=NAT_ALPHA, zorder=2)
        if b_idx + 1 in nat_bins and window in nat_bins[b_idx + 1]:
            val_next = nat_bins[b_idx + 1][window].get(metric_key, float("nan"))
            if not np.isnan(val_next):
                ax.vlines(x_hi, min(val, val_next), max(val, val_next),
                          colors=NAT_COLOR, linewidth=NAT_LW * 0.6,
                          alpha=NAT_ALPHA * 0.6, linestyle="--", zorder=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRICS = [
    ("odm_trajectory",  "off_diag_mean",
     "Global similarity  (off-diag mean sim)"),
    ("lag1_trajectory", "sim_lag1_mean",
     "Local flow  (lag-1 sim)"),
]

def main():
    nat_bins = load_natural_bins()
    gen_runs = load_generated()

    n_rows = len(WINDOWS)
    n_cols = len(METRICS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    fig.suptitle("Semantic Coherence — Global Similarity & Local Flow",
                 fontsize=13, y=1.01)

    for ri, w in enumerate(WINDOWS):
        for ci, (traj_key, scalar_key, col_label) in enumerate(METRICS):
            ax = axes[ri][ci]

            draw_staircase(ax, nat_bins, w, scalar_key)

            if traj_key is not None:
                any_gen = False
                for run in gen_runs:
                    wdata = run.get(w)
                    if wdata is None:
                        continue
                    pos  = wdata["pos_tok"]
                    vals = wdata[traj_key]
                    if not len(pos):
                        continue
                    # align: odm_trajectory has same length as msp_traj,
                    # positions start at index 1 of the window positions
                    xs, ys = smoothed_xy(pos, vals, SMOOTHING)
                    ax.plot(xs, ys, color=run["color"], lw=1.1, alpha=0.8,
                            label=run["label"] if ri == 0 and ci == 0 else "")
                    any_gen = True
                if not any_gen:
                    ax.text(0.5, 0.5, "not computed\nat this scale",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=8, color="gray", style="italic")

            ax.margins(y=0.12)
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
               loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
