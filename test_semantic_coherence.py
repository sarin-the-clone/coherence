"""
Quick test for SemanticCoherence and SemanticCoherenceMonitor.

Uses a ~40k char excerpt from a Gutenberg book (enough for several windows
at W2k and W8k) plus a synthetic looping text to verify the novelty signal.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from semantic_coherence import SemanticCoherence, SemanticCoherenceMonitor, LEVEL_WINDOW_CHARS, WINDOW_NAMES

BOOK_PATH = ROOT / "natural_texts/Selected_Books/108_The_Return_of_Sherlock_Holmes.txt"
EXCERPT_CHARS = 40_000   # enough for ~20 W2k windows and ~5 W8k windows

# ---------------------------------------------------------------------------
# Load text
# ---------------------------------------------------------------------------
with open(BOOK_PATH, encoding="utf-8", errors="replace") as f:
    full_text = f.read()

natural_excerpt = full_text[5_000 : 5_000 + EXCERPT_CHARS]
print(f"Natural excerpt: {len(natural_excerpt):,} chars")

# Synthetic looping text: same paragraph repeated many times
para = (
    "The detective stood at the window, watching the rain fall on the cobblestones. "
    "He had seen many cases in his long career, but this one troubled him more than most. "
    "The evidence pointed in contradictory directions, and the witness accounts were unreliable. "
    "He lit his pipe and considered the facts carefully, one by one.\n\n"
)
looping_text = para * (EXCERPT_CHARS // len(para) + 1)
looping_text = looping_text[:EXCERPT_CHARS]
print(f"Looping text:    {len(looping_text):,} chars")

# ---------------------------------------------------------------------------
# Test 1 — SemanticCoherence retrospective, W2k and W8k
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 1: SemanticCoherence — natural text (W2k, W8k)")
print("="*60)

sc_nat = SemanticCoherence(
    natural_excerpt,
    levels=[2, 3],
    stride_factor=0.5,
    batch_size=32,
)
res_nat = sc_nat.compute_all()

for level in [2, 3]:
    name = WINDOW_NAMES[level]
    p = f"sc_{name}_"
    n = res_nat.get(f"{p}n_windows", 0)
    print(f"\n{name} ({LEVEL_WINDOW_CHARS[level]:,} chars/window, {n} windows):")
    for key in ["off_diag_mean", "novelty_mean", "novelty_final",
                "mean_sim_to_past_final", "sim_lag1_mean"]:
        v = res_nat.get(f"{p}{key}", float("nan"))
        print(f"  {key:<28} {v:.4f}")

# ---------------------------------------------------------------------------
# Test 2 — same windows on looping text
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 2: SemanticCoherence — looping text (W2k, W8k)")
print("="*60)

sc_loop = SemanticCoherence(
    looping_text,
    levels=[2, 3],
    stride_factor=0.5,
    batch_size=32,
    tokenizer=sc_nat._tokenizer,   # reuse loaded model
    model=sc_nat._model,
)
res_loop = sc_loop.compute_all()

for level in [2, 3]:
    name = WINDOW_NAMES[level]
    p = f"sc_{name}_"
    n = res_loop.get(f"{p}n_windows", 0)
    print(f"\n{name} ({LEVEL_WINDOW_CHARS[level]:,} chars/window, {n} windows):")
    for key in ["off_diag_mean", "novelty_mean", "novelty_final",
                "mean_sim_to_past_final", "sim_lag1_mean"]:
        v = res_loop.get(f"{p}{key}", float("nan"))
        print(f"  {key:<28} {v:.4f}")

# ---------------------------------------------------------------------------
# Test 3 — SemanticCoherenceMonitor online mode
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 3: SemanticCoherenceMonitor — online mode (W2k)")
print("="*60)

monitor_nat  = SemanticCoherenceMonitor(
    level=2, loop_lag=3,
    tokenizer=sc_nat._tokenizer, model=sc_nat._model,
)
monitor_loop = SemanticCoherenceMonitor(
    level=2, loop_lag=3,
    tokenizer=sc_nat._tokenizer, model=sc_nat._model,
)

monitor_nat.update_from_text(natural_excerpt)
monitor_loop.update_from_text(looping_text)

nov_nat  = monitor_nat.novelty_trajectory()
nov_loop = monitor_loop.novelty_trajectory()

print(f"\n  Natural  — mean novelty: {np.nanmean(nov_nat):.4f}  "
      f"final: {nov_nat[-1] if len(nov_nat) else float('nan'):.4f}")
print(f"  Looping  — mean novelty: {np.nanmean(nov_loop):.4f}  "
      f"final: {nov_loop[-1] if len(nov_loop) else float('nan'):.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("EXPECTED: natural novelty > looping novelty at both levels")
print("="*60)
for level in [2, 3]:
    name = WINDOW_NAMES[level]
    p = f"sc_{name}_"
    nat_nov  = res_nat.get(f"{p}novelty_mean",  float("nan"))
    loop_nov = res_loop.get(f"{p}novelty_mean", float("nan"))
    flag = "✓" if nat_nov > loop_nov else "✗"
    print(f"  {name}: natural={nat_nov:.4f}  looping={loop_nov:.4f}  {flag}")
