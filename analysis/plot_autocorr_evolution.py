#!/usr/bin/env python3
"""
Plot autocorrelation evolution over chunks for a generated book (line plots only).

Usage:
  python analysis/plot_autocorr_evolution.py \
    --folder generated_texts_og/prompts/<book_folder> \
    --max_lag 200 \
    --out generated_texts_og/prompts/<book_folder>/autocorr_heatmap.png

What it does:
  - Loads chunk_*_attempt_*.txt (skipping *_prompt.txt) in the folder.
  - Picks the highest attempt index per chunk as the accepted text.
  - Accumulates text chunk-by-chunk (optionally every N chunks).
  - Tokenizes accumulated text (Tokenizer kind="gpt4") and computes
  normalized autocorrelation up to max_lag for each accumulation step.
  - Saves line plots of ACF curves (linear and optional log-log). Heatmap removed.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root on path so we can import tokenizer at coherence/tokenizer.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tokenizer import Tokenizer


def load_accepted_texts(folder: Path) -> List[str]:
    """
    Return list of accepted new_text per chunk (highest attempt index).
    """
    pattern = re.compile(r"chunk_(\d+)_attempt_(\d+)\.txt$")
    by_chunk: Dict[int, Dict[int, Dict]] = {}

    for file in folder.glob("chunk_*_attempt_*.txt"):
        if "prompt" in file.name:
            continue
        m = pattern.search(file.name)
        if not m:
            continue
        chunk = int(m.group(1))
        attempt = int(m.group(2))
        try:
            data = json.load(open(file))
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue
        by_chunk.setdefault(chunk, {})[attempt] = data

    texts: List[str] = []
    for chunk, attempts in sorted(by_chunk.items()):
        best_attempt = max(attempts)  # assume highest attempt accepted
        new_text = attempts[best_attempt].get("new_text", "")
        if not isinstance(new_text, str):
            new_text = str(new_text)
        texts.append(new_text)
    return texts


def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized autocorrelation up to max_lag."""
    series = series - series.mean()
    if series.std() == 0:
        return np.zeros(max_lag + 1)
    corr = np.correlate(series, series, mode="full")
    mid = len(corr) // 2
    corr = corr[mid : mid + max_lag + 1]
    return corr / corr[0] if corr[0] != 0 else np.zeros_like(corr)


def build_heatmap(texts: List[str], max_lag: int, tokenizer_kind: str = "gpt4", step: int = 1):
    """
    Return (acf_matrix, chunk_indices) where acf_matrix shape = (n_steps, max_lag+1).
    Only every `step`-th accumulation is computed; the final accumulation is always included.
    """
    tok = Tokenizer(kind=tokenizer_kind)
    acfs = []
    chunk_indices = []
    acc_text = ""
    for idx, t in enumerate(texts, start=1):
        acc_text += "\n" + t
        if idx % step != 0 and idx != len(texts):
            continue  # skip until step boundary or last chunk
        tokens = np.array(tok.encode(acc_text), dtype=float)
        if tokens.size == 0:
            acfs.append(np.zeros(max_lag + 1))
        else:
            acfs.append(autocorr(tokens, max_lag))
        chunk_indices.append(idx)
    return np.vstack(acfs), chunk_indices


def plot(acf_mat: np.ndarray, chunk_indices: List[int], out_path: Path, max_lag: int, loglog: bool):
    n_rows = 2 if loglog else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 6 if not loglog else 9),
        gridspec_kw={"height_ratios": [1, 1][:n_rows]},
        sharex=False,
    )
    if n_rows == 1:
        axes = [axes]

    ax_line = axes[0]
    lags_lin = range(max_lag + 1)
    # plot every stored step as a faint line, highlight final
    for i, row in enumerate(acf_mat):
        lbl = f"chunk {chunk_indices[i]}" if i in (0, len(acf_mat) - 1) else None
        alpha = 0.25 if i != len(acf_mat) - 1 else 0.9
        lw = 0.8 if i != len(acf_mat) - 1 else 1.5
        ax_line.plot(lags_lin, row, alpha=alpha, linewidth=lw, label=lbl, color="tab:blue")
    ax_line.set_xlabel("Lag")
    ax_line.set_ylabel("Autocorrelation")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend()

    if loglog:
        ax_ll = axes[1]
        lags = np.arange(1, max_lag + 1)
        # plot every stored step as faint, final highlighted
        for i, row in enumerate(acf_mat):
            vals = np.abs(row[1:])  # skip lag 0
            lbl = f"chunk {chunk_indices[i]}" if i in (0, len(acf_mat) - 1) else None
            alpha = 0.25 if i != len(acf_mat) - 1 else 0.9
            lw = 0.8 if i != len(acf_mat) - 1 else 1.5
            ax_ll.plot(lags, vals, label=lbl, alpha=alpha, linewidth=lw)
        ax_ll.set_xscale("log")
        ax_ll.set_yscale("log")
        ax_ll.set_xlabel("Lag (log)")
        ax_ll.set_ylabel("|ACF| (log)")
        ax_ll.grid(True, alpha=0.3, which="both")
        ax_ll.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved autocorrelation plot → {out_path}")


def plot_loglog_3d(acf_mat: np.ndarray, chunk_indices: List[int], max_lag: int, out_path: Path):
    """
    3D waterfall of log-log |ACF| for all steps (ignores --step thinning).
    X: log10(lag), Y: chunk index, Z: log10(|ACF|).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    lags = np.arange(1, max_lag + 1)
    lags_log = np.log10(lags)
    eps = 1e-9

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i, row in enumerate(acf_mat):
        vals = np.abs(row[1:]) + eps  # skip lag 0, avoid log(0)
        ax.plot(lags_log, [chunk_indices[i]] * len(lags_log), np.log10(vals), linewidth=1.0, alpha=0.7)

    ax.set_xlabel("log10(lag)")
    ax.set_ylabel("Chunk")
    ax.set_zlabel("log10(|ACF|)")
    ax.set_title("Log-log |ACF| waterfall (all iterations)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved 3D log-log ACF plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot autocorrelation evolution over chunks.")
    parser.add_argument("--folder", "-f", required=True, help="Prompt/debug folder for a generated book.")
    parser.add_argument("--max_lag", type=int, default=200, help="Maximum lag for autocorrelation.")
    parser.add_argument("--step", type=int, default=1, help="Compute ACF every N chunks (default 1 = every chunk).")
    parser.add_argument("--loglog", action="store_true", help="Add log-log |ACF| plot for final accumulation.")
    parser.add_argument("--out", "-o", default=None, help="Output image path (default: <folder>/autocorr_acf.png)")
    parser.add_argument("--loglog3d", action="store_true", help="Add 3D log-log |ACF| waterfall using all iterations.")
    parser.add_argument("--out3d", default=None, help="Output path for 3D plot (default: <folder>/autocorr_loglog3d.png)")
    parser.add_argument("--tokenizer", default="gpt4", help="Tokenizer kind (default: gpt4).")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    texts = load_accepted_texts(folder)
    if not texts:
        raise SystemExit("No chunk attempts found.")

    acf_mat, chunk_indices = build_heatmap(texts, args.max_lag, tokenizer_kind=args.tokenizer, step=args.step)
    out_path = Path(args.out) if args.out else folder / "autocorr_acf.png"
    plot(acf_mat, chunk_indices, out_path, args.max_lag, loglog=args.loglog)

    if args.loglog3d:
        # Recompute with full resolution (step=1) for 3D plot
        acf_full, chunk_full = build_heatmap(texts, args.max_lag, tokenizer_kind=args.tokenizer, step=1)
        out3d = Path(args.out3d) if args.out3d else folder / "autocorr_loglog3d.png"
        plot_loglog_3d(acf_full, chunk_full, args.max_lag, out3d)


if __name__ == "__main__":
    main()
