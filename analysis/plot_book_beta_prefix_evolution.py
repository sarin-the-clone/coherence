#!/usr/bin/env python3
"""
Plot beta evolution directly from a finalized generated book text.

This computes the autocorrelation-decay exponent beta on cumulative
prefixes of the tokenized final book text, using the project tokenizer.

Example:
  source ~/pienicolol/bin/activate
  python coherence/analysis/plot_book_beta_prefix_evolution.py \
    --book coherence/generated_texts_og/Texts/LONGBOOK__...txt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tokenizer import Tokenizer
from metrics import autocorrelation_decay


def choose_prefix_lengths(total_tokens: int, min_tokens: int, step_tokens: int) -> list[int]:
    if total_tokens <= 0:
        return []

    start = min(min_tokens, total_tokens)
    lengths = list(range(start, total_tokens + 1, step_tokens))
    if not lengths or lengths[-1] != total_tokens:
        lengths.append(total_tokens)
    return sorted(set(lengths))


def main():
    parser = argparse.ArgumentParser(description="Plot beta evolution over cumulative token prefixes of a final book text.")
    parser.add_argument("--book", required=True, help="Path to finalized book .txt file")
    parser.add_argument("--out", help="Output image path (default: alongside book with _beta_prefix_evolution.jpg suffix)")
    parser.add_argument("--min_tokens", type=int, default=20000, help="First prefix size to evaluate")
    parser.add_argument("--step_tokens", type=int, default=10000, help="Token step between evaluated prefixes")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for beta computation")
    args = parser.parse_args()

    book_path = Path(args.book).resolve()
    if not book_path.exists():
        raise SystemExit(f"Book not found: {book_path}")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = book_path.with_name(book_path.stem + "_beta_prefix_evolution.jpg")

    text = book_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer(kind="gpt4")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    if total_tokens < 10:
        raise SystemExit(f"Book is too short after tokenization: {total_tokens} tokens")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix_lengths = choose_prefix_lengths(total_tokens, args.min_tokens, args.step_tokens)
    betas = []

    print(f"Book: {book_path}")
    print(f"Total tokens: {total_tokens}")
    print(f"Device: {device}")
    print(f"Evaluating {len(prefix_lengths)} prefixes from {prefix_lengths[0]} to {prefix_lengths[-1]} tokens")

    for i, n_tok in enumerate(prefix_lengths, start=1):
        beta = autocorrelation_decay(tokens[:n_tok], device=device)
        betas.append(beta)
        if i == 1 or i == len(prefix_lengths) or i % 10 == 0:
            print(f"[{i:03d}/{len(prefix_lengths):03d}] prefix_tokens={n_tok} beta={beta}")

    xs = np.array(prefix_lengths, dtype=float)
    ys = np.array(betas, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, ys, color="tab:blue", linewidth=2.0)
    ax.scatter(xs, ys, color="tab:blue", s=10, alpha=0.7)
    ax.set_title("Beta Evolution Over Cumulative Token Prefixes")
    ax.set_xlabel("Cumulative tokens")
    ax.set_ylabel("beta")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(xs.min(), xs.max())

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
