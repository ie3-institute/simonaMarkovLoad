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


def test_cross_source_transitions_excluded():
    """Transitions between the last row of one source and the first of the next must not be counted."""
    ts = pd.to_datetime(
        ["2025-01-06 00:00", "2025-01-06 00:15", "2025-01-06 00:30", "2025-01-06 00:45"]
    )
    states = [0, 2, 5, 7]
    buckets = [bucket_id(t) for t in ts]

    # Two sources: A has rows 0-1, B has rows 2-3.
    # Valid transitions: 0→2 (within A), 5→7 (within B).
    # Spurious transition 2→5 (A→B boundary) must be excluded.
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "state": states,
            "bucket": buckets,
            "source": ["A", "A", "B", "B"],
        }
    )

    counts = build_transition_counts(df)

    b_a = buckets[0]  # FROM bucket of A's 0→2 transition (ts[0])
    b_x = buckets[1]  # FROM bucket of spurious 2→5 boundary (ts[1])
    b_b = buckets[2]  # FROM bucket of B's 5→7 transition (ts[2])

    assert counts[b_a, 0, 2] == 1  # A: 0→2
    assert counts[b_b, 5, 7] == 1  # B: 5→7
    assert counts[b_x, 2, 5] == 0  # cross-boundary must be absent
    assert counts.sum() == 2
