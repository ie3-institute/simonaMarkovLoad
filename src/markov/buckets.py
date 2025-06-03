import pandas as pd

_MONTH_FACTOR = 96 * 2
_WEEKEND_FACTOR = 96
_NUM_MONTHS = 12
_NUM_QH = 96
NUM_BUCKETS = _NUM_MONTHS * 2 * _NUM_QH  # 2 304


def _is_weekend(ts):
    if isinstance(ts, pd.Series):
        return ts.dt.dayofweek >= 5
    if isinstance(ts, pd.DatetimeIndex):
        return ts.dayofweek >= 5
    return ts.dayofweek >= 5


def bucket_id(ts):
    if isinstance(ts, pd.Series):
        weekend = _is_weekend(ts).astype(int)
        qh = ts.dt.hour * 4 + ts.dt.minute // 15
        month = ts.dt.month - 1
    elif isinstance(ts, pd.DatetimeIndex):
        weekend = _is_weekend(ts).astype(int)
        qh = ts.hour * 4 + ts.minute // 15
        month = ts.month - 1
    else:
        weekend = int(_is_weekend(ts))
        qh = ts.hour * 4 + ts.minute // 15
        month = ts.month - 1

    return month * _MONTH_FACTOR + weekend * _WEEKEND_FACTOR + qh


def assign_buckets(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    bucket_col: str = "bucket",
    inplace: bool = False,
) -> pd.DataFrame:
    tgt = df if inplace else df.copy()
    tgt[bucket_col] = bucket_id(tgt[ts_col]).astype("uint16")
    return tgt
