import numpy as np
import pandas as pd

from src.markov.buckets import NUM_BUCKETS, assign_buckets, bucket_id


def _manual_bucket(ts):
    month_factor = 96 * 2
    weekend_factor = 96
    month = ts.month - 1
    weekend = int(ts.dayofweek >= 5)
    qh = ts.hour * 4 + ts.minute // 15
    return month * month_factor + weekend * weekend_factor + qh


def test_bucket_id_single_weekday_weekend():
    ts_weekday = pd.Timestamp("2025-06-02 00:00")
    ts_weekend = pd.Timestamp("2025-06-07 00:00")

    assert bucket_id(ts_weekday) == _manual_bucket(ts_weekday)
    assert bucket_id(ts_weekend) == _manual_bucket(ts_weekend)


def test_bucket_id_series():
    s = pd.to_datetime(["2025-06-02 12:30", "2025-06-07 23:45"])
    out = bucket_id(s)
    expected = s.map(_manual_bucket)

    assert np.array_equal(out.to_numpy(), expected.to_numpy())


def test_assign_buckets_adds_column_and_range():
    ts = pd.date_range("2025-03-01", periods=4, freq="15min")
    df = pd.DataFrame({"timestamp": ts})
    out = assign_buckets(df.copy())

    assert "bucket" in out.columns
    assert out["bucket"].between(0, NUM_BUCKETS - 1).all()
