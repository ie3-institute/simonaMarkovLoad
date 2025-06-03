import matplotlib.pyplot as plt
import numpy as np

from src.markov.transition_counts import build_transition_counts
from src.markov.transitions import build_transition_matrices
from src.preprocessing.loader import load_timeseries

df = load_timeseries(normalize=True, discretize=True)


counts = build_transition_counts(df)
probs = build_transition_matrices(df, alpha=1.0)

print("counts shape :", counts.shape)
print("probs  shape :", probs.shape)


active_buckets = np.where(counts.sum(axis=(1, 2)) > 0)[0]
bucket = int(active_buckets[0]) if active_buckets.size else 0
print(f"\nUsing bucket {bucket}")


print("row sums :", probs[bucket].sum(axis=1))

plt.imshow(probs[bucket], aspect="auto")
plt.title(f"Bucket {bucket} – transition probabilities")
plt.xlabel("state t+1")
plt.ylabel("state t")
plt.colorbar()
plt.tight_layout()
plt.show()
