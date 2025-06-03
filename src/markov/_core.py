import numpy as np
import pandas as pd

from src.config import CONFIG

from .buckets import NUM_BUCKETS

N_STATES = int(CONFIG["model"]["n_states"])


def _transition_counts(
    df: pd.DataFrame, *, state_col="state", bucket_col="bucket", dtype=np.uint32
) -> np.ndarray:
    df = df.sort_values("timestamp")
    s_t = df[state_col].to_numpy(dtype=int)[:-1]
    s_tp1 = df[state_col].to_numpy(dtype=int)[1:]
    buckets = df[bucket_col].to_numpy(dtype=int)[:-1]

    counts = np.zeros((NUM_BUCKETS, N_STATES, N_STATES), dtype=dtype)
    np.add.at(counts, (buckets, s_t, s_tp1), 1)
    return counts
