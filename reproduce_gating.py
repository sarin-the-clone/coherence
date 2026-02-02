
import sys
import os

# Ensure we can import the module
sys.path.append(os.getcwd())

from generated_texts_og.utils.coherence_decision import get_dynamic_thresholds, _BINNED_CFG

print("Loaded Config Bins:", len(_BINNED_CFG["bins"]) if _BINNED_CFG else "None")

# Test cases
tokens_to_test = [100000, 200000, 400_000, 1_000_000]

for t in tokens_to_test:
    print(f"\n--- Testing with {t} tokens ---")
    cfg = get_dynamic_thresholds(t)
    print(f"Beta Max: {cfg.get('beta_max')}")
    print(f"Alpha Max: {cfg.get('alpha_max')}")
    print(f"Bin Index Logic Check:")
    if t < 150000:
        print("  Expected Bin: 0")
    elif t < 300000:
        print("  Expected Bin: 1")
    else:
        print("  Expected Bin: 2")
