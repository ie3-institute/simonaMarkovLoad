import pandas as pd

from src.markov.buckets import NUM_BUCKETS, bucket_id
from src.markov.transition_counts import N_STATES, build_transition_counts


def test_build_transition_counts_basic():
    ts = pd.to_datetime(["2025-01-06 00:00", "2025-01-06 00:15", "2025-01-06 00:30"])
    states = [0, 1, 1]
    buckets = [bucket_id(t) for t in ts]

    df = pd.DataFrame({"timestamp": ts, "state": states, "bucket": buckets})

    counts = build_transition_counts(df)

    assert counts.shape == (NUM_BUCKETS, N_STATES, N_STATES)

    assert counts[buckets[0], 0, 1] == 1
    assert counts[buckets[1], 1, 1] == 1

    assert counts.sum() == 2
