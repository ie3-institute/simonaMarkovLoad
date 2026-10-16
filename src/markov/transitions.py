import numpy as np
import pandas as pd

from .transition_counts import build_transition_counts


def build_transition_matrices(
    df: pd.DataFrame,
    *,
    counts: np.ndarray | None = None,
    dtype=np.float32,
) -> np.ndarray:
    if counts is None:
        counts = build_transition_counts(df, dtype=np.uint32)
    counts = counts.astype(np.float64)
    empty = counts.sum(axis=2) == 0
    if np.any(empty):
        # states without data in a bucket move to the states reached in this bucket,
        # so that they can't trap the chain; buckets without any data keep a self-loop
        reached = counts.sum(axis=1)
        idx_b, idx_i = np.where(empty)
        counts[idx_b, idx_i, :] = reached[idx_b]
        no_data = reached[idx_b].sum(axis=1) == 0
        counts[idx_b[no_data], idx_i[no_data], idx_i[no_data]] = 1
    probs = counts / counts.sum(axis=2, keepdims=True)

    return probs.astype(dtype)
