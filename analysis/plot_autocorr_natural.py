#!/usr/bin/env python3
"""
Plot autocorrelation for a natural (non-iterative) book.

Usage:
  python analysis/plot_autocorr_natural.py --text /path/to/book.txt --max_lag 200 \
    --out /path/to/output.png [--loglog]

Notes:
  - Uses the shared gpt4-based tokenizer (coherence/tokenizer.py) for consistency with generated books.
  - If you need a Gutenberg text, download it first (e.g., via coherence/natural_texts/retrieve_gutenberg.py)
    and point --text to the plain .txt file.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# Ensure repo root on path for tokenizer import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tokenizer import Tokenizer


def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized autocorrelation up to max_lag."""
    series = series - series.mean()
    if series.std() == 0:
        return np.zeros(max_lag + 1)
    corr = np.correlate(series, series, mode="full")
    mid = len(corr) // 2
    corr = corr[mid : mid + max_lag + 1]
    return corr / corr[0] if corr[0] != 0 else np.zeros_like(corr)


def plot_acf(acf: np.ndarray, max_lag: int, out_path: Path, loglog: bool):
    n_rows = 2 if loglog else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 6 if not loglog else 9),
        gridspec_kw={"height_ratios": [1, 1][:n_rows]},
    )
    if n_rows == 1:
        axes = [axes]

    lags_lin = range(max_lag + 1)
    ax_lin = axes[0]
    ax_lin.plot(lags_lin, acf, label="ACF")
    ax_lin.set_xlabel("Lag (tokens)")
    ax_lin.set_ylabel("Autocorrelation")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend()

    if loglog:
        ax_ll = axes[1]
        lags = np.arange(1, max_lag + 1)
        vals = np.abs(acf[1:])  # skip lag 0
        ax_ll.plot(lags, vals, label="|ACF|")
        ax_ll.set_xscale("log")
        ax_ll.set_yscale("log")
        ax_ll.set_xlabel("Lag (log)")
        ax_ll.set_ylabel("|ACF| (log)")
        ax_ll.grid(True, alpha=0.3, which="both")
        ax_ll.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved ACF plot → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot autocorrelation for a single (natural) book.")
    ap.add_argument("--text", "-t", help="Path to plain text file.")
    ap.add_argument("--gutenberg-id", type=int, help="Gutenberg book ID to download and plot.")
    ap.add_argument("--max_lag", type=int, default=200, help="Maximum lag for autocorrelation.")
    ap.add_argument("--loglog", action="store_true", help="Add log-log |ACF| subplot.")
    ap.add_argument("--out", "-o", default=None, help="Output image path (default: alongside text as *_acf.png)")
    ap.add_argument("--tokenizer", default="gpt4", help="Tokenizer kind (default: gpt4).")
    args = ap.parse_args()

    if not args.text and not args.gutenberg_id:
        raise SystemExit("Provide --text or --gutenberg-id")

    # Retrieve text
    if args.gutenberg_id:
        gid = args.gutenberg_id
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        print(f"📥 Downloading Gutenberg ID {gid} …")
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.text) < 1000:
            raise SystemExit(f"Failed to download Gutenberg ID {gid}")
        raw_text = r.text
        # Trim boilerplate if markers exist
        start = raw_text.find("*** START")
        end = raw_text.find("*** END")
        if start != -1 and end != -1:
            text = raw_text[start:end].strip()
        else:
            text = raw_text.strip()
        text_path = Path(f"gutenberg_{gid}.txt")
    else:
        text_path = Path(args.text).expanduser()
        if not text_path.exists():
            raise SystemExit(f"Text file not found: {text_path}")
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    tok = Tokenizer(kind=args.tokenizer)
    tokens = np.array(tok.encode(text), dtype=float)
    if tokens.size == 0:
        raise SystemExit("No tokens found in text.")

    acf = autocorr(tokens, args.max_lag)
    out_path = Path(args.out) if args.out else text_path.with_name(text_path.stem + "_acf.png")
    plot_acf(acf, args.max_lag, out_path, args.loglog)


if __name__ == "__main__":
    main()
