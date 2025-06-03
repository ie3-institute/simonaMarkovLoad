"""Time-bucket mapping: 12 months × weekend flag × 96 quarter-hours."""

from __future__ import annotations

import pandas as pd

_MONTH_FACTOR = 96 * 2  # offset between months
_WEEKEND_FACTOR = 96  # offset between weekday/weekend
_NUM_MONTHS = 12
_NUM_QH = 96  # 24 h × 4
NUM_BUCKETS = _NUM_MONTHS * 2 * _NUM_QH  # 2 304


def _is_weekend(ts):
    """Return bool array/scalar: Saturday or Sunday ⇒ True."""
    if isinstance(ts, pd.Series):
        return ts.dt.dayofweek >= 5
    if isinstance(ts, pd.DatetimeIndex):
        return ts.dayofweek >= 5
    return ts.dayofweek >= 5


def bucket_id(ts):
    """Vectorised/ scalar timestamp → unique bucket integer (0…2303)."""
    if isinstance(ts, pd.Series):
        weekend = _is_weekend(ts).astype(int)
        qh = ts.dt.hour * 4 + ts.dt.minute // 15
        month = ts.dt.month - 1
    elif isinstance(ts, pd.DatetimeIndex):
        weekend = _is_weekend(ts).astype(int)
        qh = ts.hour * 4 + ts.minute // 15
        month = ts.month - 1
    else:  # single Timestamp
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
    """Add a *bucket* column (uint16 recommended)."""
    tgt = df if inplace else df.copy()
    tgt[bucket_col] = bucket_id(tgt[ts_col]).astype("uint16")
    return tgt
