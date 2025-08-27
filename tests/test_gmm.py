import numpy as np
import pandas as pd

from src.config import CONFIG
from src.markov.buckets import NUM_BUCKETS, assign_buckets
from src.markov.gmm import fit_gmms, sample_value


def test_gmm_structure(tiny_models):
    _, gmms = tiny_models

    assert len(gmms) == NUM_BUCKETS

    n_states = CONFIG["model"]["n_states"]

    for _bucket_idx, bucket_states in enumerate(gmms):
        assert len(bucket_states) == n_states

        for _state_idx, gmm in enumerate(bucket_states):
            if gmm is not None:
                assert isinstance(gmm, tuple | list)
                assert len(gmm) == 3

                weights, means, variances = gmm

                assert isinstance(weights, np.ndarray)
                assert isinstance(means, np.ndarray)
                assert isinstance(variances, np.ndarray)

                assert len(weights) == len(means) == len(variances)

                assert np.allclose(weights.sum(), 1.0, atol=1e-6)

                assert np.all(weights >= 0)

                assert np.all(variances >= 0)

                assert np.all(means >= 0) and np.all(means <= 1)


def test_gmm_fit_recovery(rng):
    n_samples = 500

    component_1_samples = int(0.6 * n_samples)
    component_2_samples = n_samples - component_1_samples

    data_1 = rng.normal(0.2, 0.005, component_1_samples)
    data_2 = rng.normal(0.7, 0.01, component_2_samples)

    all_data = np.concatenate([data_1, data_2])
    rng.shuffle(all_data)

    all_data = np.clip(all_data, 0.01, 0.99)

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="15min")
    df = pd.DataFrame(
        {"timestamp": timestamps, "value": all_data, "state": [0] * n_samples}
    )
    df = assign_buckets(df, ts_col="timestamp")

    gmms = fit_gmms(
        df,
        value_col="value",
        k_candidates=(1, 2, 3),
        min_samples=10,
        random_state=123,
        verbose=0,
        n_jobs=1,
    )

    bucket = df["bucket"].iloc[0]
    state = 0
    fitted_gmm = gmms[bucket][state]

    assert fitted_gmm is not None

    weights, means, variances = fitted_gmm
    n_components = len(weights)

    assert n_components in [1, 2]

    if n_components == 2:
        sorted_indices = np.argsort(means)
        sorted_means = means[sorted_indices]
        sorted_weights = weights[sorted_indices]

        assert np.abs(sorted_means[0] - 0.2) < 0.05
        assert np.abs(sorted_means[1] - 0.7) < 0.05

        assert np.abs(sorted_weights[0] - 0.6) < 0.15
        assert np.abs(sorted_weights[1] - 0.4) < 0.15


def test_sample_value_deterministic(tiny_models, rng):
    _, gmms = tiny_models

    populated_gmm = None
    bucket_idx, state_idx = None, None

    for b, bucket_states in enumerate(gmms):
        for s, gmm in enumerate(bucket_states):
            if gmm is not None:
                populated_gmm = gmm
                bucket_idx, state_idx = b, s
                break
        if populated_gmm is not None:
            break

    assert populated_gmm is not None

    test_rng = np.random.default_rng(456)
    samples = [
        sample_value(gmms, bucket_idx, state_idx, rng=test_rng) for _ in range(10)
    ]

    test_rng = np.random.default_rng(456)
    samples_repeat = [
        sample_value(gmms, bucket_idx, state_idx, rng=test_rng) for _ in range(10)
    ]

    assert np.allclose(samples, samples_repeat)

    assert all(0 <= s <= 1 for s in samples)

    weights, means, variances = populated_gmm
    if len(weights) > 1 or variances[0] > 1e-6:
        assert not all(s == samples[0] for s in samples)


def test_gmm_empty_state():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": [0.5, 0.5, 0.5, 0.5, 0.5],
            "state": [0, 0, 0, 0, 0],
        }
    )
    df = assign_buckets(df, ts_col="timestamp")

    gmms = fit_gmms(
        df,
        value_col="value",
        min_samples=1,
        k_candidates=(1,),
        random_state=123,
        verbose=0,
        n_jobs=1,
    )

    n_states = CONFIG["model"]["n_states"]
    populated_bucket = df["bucket"].iloc[0]

    assert gmms[populated_bucket][0] is not None

    for state in range(1, n_states):
        assert gmms[populated_bucket][state] is None

    empty_bucket = (populated_bucket + 1) % NUM_BUCKETS
    for state in range(n_states):
        assert gmms[empty_bucket][state] is None
