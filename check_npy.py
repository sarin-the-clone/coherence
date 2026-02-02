import numpy as np
import sys

path = "generated_texts_og/prompts/20251217_052836/memory_chunk_068/tokens.npy"
try:
    arr = np.load(path)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"First 10: {arr[:10]}")
    print(f"Last 10: {arr[-10:]}")
    print(f"Unique: {len(np.unique(arr))}")
    
    # Check for zeros
    zeros = np.sum(arr == 0)
    print(f"Zeros: {zeros}")
    
    # Check for repetition
    # Simple check: is the second half equal to first half?
    mid = len(arr) // 2
    if np.array_equal(arr[:mid], arr[mid:2*mid]):
        print("DETECTED EXACT DUPLICATION (x2)")
    
except Exception as e:
    print(e)
