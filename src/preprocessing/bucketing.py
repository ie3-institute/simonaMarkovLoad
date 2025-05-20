from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

QUARTERS_PER_DAY: int = 96
WEEKEND_BUCKETS_PER_MONTH: int = 2 * QUARTERS_PER_DAY  # 192
TOTAL_BUCKETS: int = 12 * WEEKEND_BUCKETS_PER_MONTH  # 2_304


@dataclass(frozen=True, slots=True)
class Bucket:

    month: int
    weekend: int
    quarter: int

    def key(self) -> Tuple[int, int, int]:

        return (self.month, self.weekend, self.quarter)

    def index(self) -> int:

        return (
            (self.month - 1) * WEEKEND_BUCKETS_PER_MONTH
            + self.weekend * QUARTERS_PER_DAY
            + self.quarter
        )

    @classmethod
    def from_timestamp(cls, ts: pd.Timestamp) -> "Bucket":

        return cls(
            month=ts.month,
            weekend=int(ts.day_of_week >= 5),
            quarter=ts.hour * 4 + ts.minute // 15,
        )


def _timestamp_series(
    df: pd.DataFrame, *, timestamp_col: str = "timestamp"
) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        return pd.to_datetime(df[timestamp_col], utc=False, infer_datetime_format=True)
    return df[timestamp_col]


def add_bucket_columns(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    inplace: bool = False,
) -> pd.DataFrame:

    base = df if inplace else df.copy()

    ts = _timestamp_series(base, timestamp_col=timestamp_col)

    base["month"] = ts.dt.month
    base["weekend"] = (ts.dt.dayofweek >= 5).astype("uint8")
    base["quarter"] = (ts.dt.hour * 4 + ts.dt.minute // 15).astype("uint16")

    base["bucket_id"] = (
        (base["month"] - 1) * WEEKEND_BUCKETS_PER_MONTH
        + base["weekend"] * QUARTERS_PER_DAY
        + base["quarter"]
    ).astype("uint16")

    return base


__all__ = [
    "Bucket",
    "add_bucket_columns",
    "QUARTERS_PER_DAY",
    "WEEKEND_BUCKETS_PER_MONTH",
    "TOTAL_BUCKETS",
]
