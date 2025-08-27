import numpy as np
import pandas as pd
from ._core import _transition_counts



DEFAULT_MIN_COUNT   = 25
DEFAULT_UNIFORM_EPS = 0.25


def build_transition_matrices(
    df: pd.DataFrame,
    *,
    dtype: np.dtype = np.float32,
    min_count: int = DEFAULT_MIN_COUNT,
    uniform_eps: float = DEFAULT_UNIFORM_EPS,
) -> np.ndarray:


    counts: np.ndarray = _transition_counts(df, dtype=np.float64)
    _, n_states, _ = counts.shape

    row_sum = counts.sum(axis=2, keepdims=True)
    mask_empty  = (row_sum == 0)
    mask_sparse = (row_sum < min_count) & ~mask_empty

    if mask_empty.any():
        counts[mask_empty.repeat(n_states, axis=2)] = 1.0

    if mask_sparse.any():
        counts[mask_sparse.repeat(n_states, axis=2)] += uniform_eps / n_states

    row_sum = counts.sum(axis=2, keepdims=True)
    probs   = counts / row_sum

    return probs.astype(dtype)
