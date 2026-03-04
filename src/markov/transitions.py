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
    counts = counts.copy()
    row_sum = counts.sum(axis=2, keepdims=True)
    empty = row_sum == 0
    if np.any(empty):
        idx_b, idx_i, _ = np.where(empty)
        counts[idx_b, idx_i, :] = 0
        counts[idx_b, idx_i, idx_i] = 1
    row_sum[empty] = 1
    probs = counts / row_sum

    return probs.astype(dtype)
