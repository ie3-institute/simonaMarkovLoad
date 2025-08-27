import numpy as np
import pandas as pd
import pytest

from src.markov.buckets import assign_buckets
from src.markov.gmm import fit_gmms
from src.markov.transitions import build_transition_matrices
from src.preprocessing.scaling import discretize_states


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def small_df(rng):
    start_date = pd.Timestamp("2024-01-01 00:00:00")
    periods = 8 * 7 * 24 * 4

    timestamps = pd.date_range(start_date, periods=periods, freq="15min")

    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek

    base_load = 0.3 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24)
    morning_peak = 0.15 * np.exp(-(((hour_of_day - 8) / 2) ** 2))
    evening_peak = 0.2 * np.exp(-(((hour_of_day - 20) / 2) ** 2))

    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
    weekend_shift = np.where(
        day_of_week >= 5, 0.1 * np.sin(2 * np.pi * (hour_of_day - 2) / 24), 0
    )

    value = (base_load + morning_peak + evening_peak) * weekend_factor + weekend_shift

    noise = rng.normal(0, 0.05, len(timestamps))
    value = np.clip(value + noise, 0.05, 0.95)

    df = pd.DataFrame(
        {
            "ts": timestamps,
            "timestamp": timestamps,
            "value": value,
            "month": timestamps.month - 1,
            "is_weekend": timestamps.dayofweek >= 5,
            "quarter_hour": timestamps.hour * 4 + timestamps.minute // 15,
        }
    )

    df = assign_buckets(df, ts_col="timestamp")
    df = discretize_states(df, value_col="value")

    return df


@pytest.fixture
def tiny_models(small_df):
    P = build_transition_matrices(small_df)

    gmms = fit_gmms(
        small_df,
        value_col="value",
        min_samples=5,
        k_candidates=(1, 2),
        n_jobs=1,
        random_state=123,
        verbose=0,
    )

    return P, gmms
