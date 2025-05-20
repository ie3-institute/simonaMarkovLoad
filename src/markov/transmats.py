from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.preprocessing.bucketing import TOTAL_BUCKETS


def _counts_for_bucket(
    bid: int, bucket_ids: np.ndarray, idx_flat: np.ndarray, n_states: int
) -> Tuple[int, np.ndarray]:

    rows = idx_flat[bucket_ids == bid]
    C = (
        np.bincount(rows, minlength=n_states * n_states)
        .astype(np.uint32)
        .reshape(n_states, n_states)
    )
    return bid, C


def build_transition_matrices_parallel(
    df: pd.DataFrame,
    *,
    bucket_col: str = "bucket_id",
    state_col: str = "state",
    n_states: int,
    alpha: float = 0.5,
    n_jobs: int | None = -1,
) -> Tuple[np.ndarray, np.ndarray]:

    df = df.sort_values("timestamp")
    df["next_state"] = df[state_col].shift(-1)
    df["next_bucket"] = df[bucket_col].shift(-1)

    mask = df[bucket_col] == df["next_bucket"]
    pairs = df.loc[mask, [bucket_col, state_col, "next_state"]].to_numpy(np.uint16)

    bucket_ids = pairs[:, 0]
    idx_flat = pairs[:, 1] * n_states + pairs[:, 2]

    unique_buckets = np.arange(TOTAL_BUCKETS, dtype=np.uint16)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_counts_for_bucket)(bid, bucket_ids, idx_flat, n_states)
        for bid in unique_buckets
    )

    counts = np.zeros((TOTAL_BUCKETS, n_states, n_states), dtype=np.uint32)
    for bid, C in results:
        counts[bid] = C

    counts_sm = counts.astype(np.float32) + alpha
    row_sums = counts_sm.sum(axis=2, keepdims=True)
    probs = (counts_sm / row_sums).astype(np.float32)

    return counts, probs
