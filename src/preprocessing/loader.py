from pathlib import Path
from typing import List

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

__all__ = ["load_raw_timeseries", "RAW_DATA_DIR"]


def load_raw_timeseries(
    *,
    value_dtype: str = "float32",
) -> pd.DataFrame:

    # Collect all .csv files under data/raw
    csv_files: List[Path] = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}.")

    # Read columns (ts, value) from each file
    frames = [
        pd.read_csv(
            path,
            usecols=["ts", "value"],
            dtype={"value": value_dtype},
            parse_dates=["ts"],
        )
        for path in csv_files
    ]

    df = pd.concat(frames, ignore_index=True)

    # Rename columns to the meaningful names
    df = df.rename(columns={"ts": "timestamp", "value": "power"})

    return df
