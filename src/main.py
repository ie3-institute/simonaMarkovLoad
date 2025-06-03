# test_markov.py
# ────────────────────────────────────────────────────────────────
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.markov.transition_counts import build_transition_counts
from src.markov.transitions import build_transition_matrices

# your project imports
from src.preprocessing.loader import load_timeseries

# 1) load + preprocess  (→ columns: timestamp, state, bucket)
df = load_timeseries(
    normalize=True, discretize=True
)  # uses the default CSV in data/raw

# 2) build raw 10×10 count tensors  &  Laplace-smoothed probabilities
counts = build_transition_counts(df)  # uint32, shape (2304, 10, 10)
probs = build_transition_matrices(df, alpha=1.0)  # float32, same shape

print("counts shape :", counts.shape)
print("probs  shape :", probs.shape)

# 3) pick first bucket that actually has data
active_buckets = np.where(counts.sum(axis=(1, 2)) > 0)[0]
bucket = int(active_buckets[0]) if active_buckets.size else 0
print(f"\nUsing bucket {bucket}")

# sanity-check: rows should sum to 1
print("row sums :", probs[bucket].sum(axis=1))

# 4) quick heat-map
plt.imshow(probs[bucket], aspect="auto")
plt.title(f"Bucket {bucket} – transition probabilities")
plt.xlabel("state t+1")
plt.ylabel("state t")
plt.colorbar()
plt.tight_layout()
plt.show()

# 5) optional: save matrices for later inspection
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "transition_counts.npy", counts)
np.save(out / "transition_matrices.npy", probs)
print(f"\nSaved .npy files to {out.resolve()}")
