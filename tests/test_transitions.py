import numpy as np
import pandas as pd

from src.config import CONFIG
from src.markov.buckets import NUM_BUCKETS, assign_buckets
from src.markov.transition_counts import build_transition_counts
from src.markov.transitions import build_transition_matrices


def test_shape(tiny_models):
    p, _ = tiny_models
    n_states = CONFIG["model"]["n_states"]

    assert p.shape == (NUM_BUCKETS, n_states, n_states)
    assert p.shape == (2304, n_states, n_states)


def test_rows_sum_to_one(tiny_models, rng):
    p, _ = tiny_models
    n_buckets, n_states, _ = p.shape

    test_pairs = [
        (rng.integers(0, n_buckets), rng.integers(0, n_states)) for _ in range(20)
    ]

    for b, i in test_pairs:
        row_sum = p[b, i, :].sum()
        assert np.allclose(row_sum, 1.0, atol=1e-6)

    assert not np.any(np.isnan(p))
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_empty_row_self_loop():
    n_states = CONFIG["model"]["n_states"]

    timestamps = pd.date_range("2024-01-01", periods=10, freq="15min")
    df = pd.DataFrame({"timestamp": timestamps, "value": [0.5] * 10, "state": [0] * 10})

    df = assign_buckets(df, ts_col="timestamp")
    p = build_transition_matrices(df)

    populated_bucket = df["bucket"].iloc[0]

    assert np.allclose(p[populated_bucket, 0, 0], 1.0, atol=1e-6)

    for state in range(1, n_states):
        assert np.allclose(p[populated_bucket, state, state], 1.0, atol=1e-6)

    empty_buckets = [b for b in range(NUM_BUCKETS) if b != populated_bucket][:10]
    for bucket in empty_buckets:
        for state in range(n_states):
            expected_row = np.zeros(n_states)
            expected_row[state] = 1.0
            assert np.allclose(p[bucket, state, :], expected_row, atol=1e-6)


def test_transition_counts_structure():
    timestamps = pd.date_range("2024-01-01 00:00", periods=5, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": [0.1, 0.3, 0.7, 0.9, 0.5],
            "state": [0, 1, 2, 3, 1],
        }
    )
    df = assign_buckets(df, ts_col="timestamp")

    counts = build_transition_counts(df)
    n_states = CONFIG["model"]["n_states"]

    assert counts.shape == (NUM_BUCKETS, n_states, n_states)
    assert counts.dtype in [np.uint32, np.int32, np.int64]
    assert np.all(counts >= 0)

    total_transitions = counts.sum()
    expected_transitions = len(df) - 1
    assert total_transitions == expected_transitions


def test_build_transition_matrices_row_sums():
    from src.markov.buckets import bucket_id
    from src.markov.transition_counts import N_STATES

    ts = pd.to_datetime(["2025-02-03 00:00", "2025-02-03 00:15", "2025-02-03 00:30"])
    states = [2, 3, 2]
    buckets = [bucket_id(t) for t in ts]

    df = pd.DataFrame({"timestamp": ts, "state": states, "bucket": buckets})

    probs = build_transition_matrices(df)

    assert probs.shape == (NUM_BUCKETS, N_STATES, N_STATES)
    assert np.allclose(probs.sum(axis=2), 1.0, atol=1e-6)
