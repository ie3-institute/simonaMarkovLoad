import numpy as np
import pytest

from src.markov.buckets import bucket_id
from src.markov.gmm import sample_value


def simulate_step(p, gmms, bucket, state, rng):
    next_state = rng.choice(p.shape[1], p=p[bucket, state, :])
    sampled_value = sample_value(gmms, bucket, next_state, rng=rng)
    return next_state, sampled_value


def test_simulate_step_deterministic(tiny_models, rng):
    p, gmms = tiny_models

    test_bucket, test_state = None, None

    for bucket in range(min(100, p.shape[0])):
        for state in range(p.shape[1]):
            if gmms[bucket][state] is not None:
                transitions = p[bucket, state, :]
                if np.sum(transitions > 0.1) > 1:
                    test_bucket, test_state = bucket, state
                    break
        if test_bucket is not None:
            break

    if test_bucket is None:
        for bucket in range(p.shape[0]):
            for state in range(p.shape[1]):
                if gmms[bucket][state] is not None:
                    test_bucket, test_state = bucket, state
                    break
            if test_bucket is not None:
                break

    assert test_bucket is not None

    test_rng_1 = np.random.default_rng(789)
    result_1 = simulate_step(p, gmms, test_bucket, test_state, test_rng_1)

    test_rng_2 = np.random.default_rng(789)
    result_2 = simulate_step(p, gmms, test_bucket, test_state, test_rng_2)

    assert result_1 == result_2

    next_state, value = result_1
    assert isinstance(next_state, (int, np.integer))
    assert 0 <= next_state < p.shape[1]
    assert isinstance(value, (float, np.floating))
    assert 0 <= value <= 1


def test_multi_step_simulation_properties(tiny_models, rng):
    p, gmms = tiny_models

    start_bucket, start_state = None, None
    for bucket in range(min(50, p.shape[0])):
        for state in range(p.shape[1]):
            if gmms[bucket][state] is not None:
                start_bucket, start_state = bucket, state
                break
        if start_bucket is not None:
            break

    assert start_bucket is not None

    n_steps = 200
    states_visited = []
    values_sampled = []

    current_state = start_state
    test_rng = np.random.default_rng(456)

    for step in range(n_steps):
        current_bucket = (start_bucket + step // 10) % min(100, p.shape[0])

        if gmms[current_bucket][current_state] is None:
            current_bucket = start_bucket

        next_state, value = simulate_step(
            p, gmms, current_bucket, current_state, test_rng
        )

        states_visited.append(next_state)
        values_sampled.append(value)
        current_state = next_state

    assert all(0 <= v <= 1 for v in values_sampled)

    unique_values = len(set(np.round(values_sampled, decimals=6)))
    assert unique_values > 1

    unique_states = len(set(states_visited))
    assert unique_states > 1

    for state in states_visited:
        assert 0 <= state < p.shape[1]

    mean_value = np.mean(values_sampled)
    std_value = np.std(values_sampled)

    assert 0 < mean_value < 1
    assert std_value > 0


def test_simulate_step_state_transitions(tiny_models, rng):
    p, gmms = tiny_models

    test_cases = []

    for bucket in range(min(50, p.shape[0])):
        for state in range(p.shape[1]):
            if gmms[bucket][state] is not None:
                transitions = p[bucket, state, :]
                if np.sum(transitions > 0.01) >= 2:
                    test_cases.append((bucket, state, transitions))
                    if len(test_cases) >= 3:
                        break
        if len(test_cases) >= 3:
            break

    if not test_cases:
        pytest.skip("No suitable test cases found with non-trivial transitions")

    bucket, state, expected_transitions = test_cases[0]

    n_trials = 1000
    next_states = []

    test_rng = np.random.default_rng(999)

    for _ in range(n_trials):
        next_state, _ = simulate_step(p, gmms, bucket, state, test_rng)
        next_states.append(next_state)

    unique_states, counts = np.unique(next_states, return_counts=True)
    empirical_probs = counts / n_trials

    for i, unique_state in enumerate(unique_states):
        expected_prob = expected_transitions[unique_state]
        empirical_prob = empirical_probs[i]

        tolerance = 0.05
        assert abs(empirical_prob - expected_prob) < tolerance


def test_bucket_time_consistency():
    import pandas as pd

    test_times = [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 12:00:00"),
        pd.Timestamp("2024-01-06 18:30:00"),
        pd.Timestamp("2024-06-15 09:15:00"),
        pd.Timestamp("2024-12-31 23:45:00"),
    ]

    for ts in test_times:
        bucket = bucket_id(ts)

        assert 0 <= bucket < 2304

        month = ts.month - 1
        is_weekend = int(ts.dayofweek >= 5)
        quarter_hour = ts.hour * 4 + ts.minute // 15

        expected_bucket = month * 192 + is_weekend * 96 + quarter_hour
        assert bucket == expected_bucket

        assert 0 <= month <= 11
        assert 0 <= is_weekend <= 1
        assert 0 <= quarter_hour <= 95
