from pathlib import Path
from typing import List

import pandas as pd

from .scaling import discretize_power, normalize_power

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

__all__ = ["load_timeseries", "RAW_DATA_DIR"]


def load_timeseries(
    *,
    value_dtype: str = "float32",
    normalize: bool = False,
    discretize: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:

    # Collect all .csv files under data/raw
    csv_files: List[Path] = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}.")

    # Read columns (ts, value) from each file
    frames: List[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(
            path,
            skiprows=21,
            usecols=["Zeitstempel", "Messwert"],
            dtype={"Messwert": value_dtype},
            parse_dates=["Zeitstempel"],
        )
        df = df.rename(columns={"Zeitstempel": "timestamp", "Messwert": "cum_kwh"})

        df["power"] = df["cum_kwh"].diff() * 4

        df = df.dropna(subset=["power"]).drop(columns="cum_kwh").reset_index(drop=True)

        if normalize:
            df = normalize_power(df, col="power", eps=eps)

        if discretize:
            df = discretize_power(df, col="power", state_col="state")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)
