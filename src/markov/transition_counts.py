import numpy as np
import pandas as pd

from src.config import CONFIG

from .buckets import NUM_BUCKETS

N_STATES = int(CONFIG["model"]["n_states"])


def build_transition_counts(
    df: pd.DataFrame,
    *,
    state_col: str = "state",
    bucket_col: str = "bucket",
    source_col: str = "source",
    dtype=np.uint32,
) -> np.ndarray:
    sort_cols = [source_col, "timestamp"] if source_col in df.columns else ["timestamp"]
    df = df.sort_values(sort_cols)

    s_t = df[state_col].to_numpy(dtype=int)[:-1]
    s_tp1 = df[state_col].to_numpy(dtype=int)[1:]
    buckets = df[bucket_col].to_numpy(dtype=int)[:-1]

    if source_col in df.columns:
        sources = df[source_col].to_numpy()
        valid = sources[:-1] == sources[1:]
        s_t = s_t[valid]
        s_tp1 = s_tp1[valid]
        buckets = buckets[valid]

    counts = np.zeros((NUM_BUCKETS, N_STATES, N_STATES), dtype=dtype)
    np.add.at(counts, (buckets, s_t, s_tp1), 1)
    return counts
