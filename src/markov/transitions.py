import numpy as np
import pandas as pd

from ._core import _transition_counts


def build_transition_matrices(df: pd.DataFrame, *, dtype=np.float32) -> np.ndarray:
    # Accumulate counts using an integer dtype to avoid floating-point drift
    counts = _transition_counts(df, dtype=np.uint32)
    row_sum = counts.sum(axis=2, keepdims=True)
    empty = row_sum == 0
    if np.any(empty):
        idx_b, idx_i, _ = np.where(empty)
        counts[idx_b, idx_i, :] = 0
        counts[idx_b, idx_i, idx_i] = 1
    row_sum[empty] = 1
    probs = counts / row_sum

    return probs.astype(dtype)
