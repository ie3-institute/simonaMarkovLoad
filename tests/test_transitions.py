import numpy as np
import pandas as pd

from src.markov.buckets import NUM_BUCKETS, bucket_id
from src.markov.transition_counts import N_STATES
from src.markov.transitions import build_transition_matrices


def test_build_transition_matrices_row_sums():
    ts = pd.to_datetime(["2025-02-03 00:00", "2025-02-03 00:15", "2025-02-03 00:30"])
    states = [2, 3, 2]
    buckets = [bucket_id(t) for t in ts]

    df = pd.DataFrame({"timestamp": ts, "state": states, "bucket": buckets})

    probs = build_transition_matrices(df)

    assert probs.shape == (NUM_BUCKETS, N_STATES, N_STATES)

    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-6)
