import math
from pathlib import Path

import pandas as pd

from src.config import CONFIG

from ..markov.buckets import assign_buckets
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
    cfg_in = CONFIG["input"]

    # Collect all .csv files under data/raw
    csv_files: list[Path] = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}.")

    # Read columns (ts, value) from each file
    frames: list[pd.DataFrame] = []
    global_min = math.inf
    global_max = -math.inf
    dropped_negative_deltas = 0
    for path in csv_files:
        df = pd.read_csv(
            path,
            skiprows=cfg_in["skiprows"],
            usecols=[cfg_in["timestamp_col"], cfg_in["value_col"]],
            dtype={cfg_in["value_col"]: value_dtype},
            parse_dates=[cfg_in["timestamp_col"]],
        )
        df = df.rename(
            columns={
                cfg_in["timestamp_col"]: "timestamp",
                cfg_in["value_col"]: "cum_kwh",
            }
        )

        df["power"] = df["cum_kwh"].diff() * cfg_in["factor"]

        if cfg_in.get("drop_negative_deltas", True):
            negative_delta_mask = df["power"] < 0
            dropped_negative_deltas += int(negative_delta_mask.sum())
            df = df.loc[~negative_delta_mask]

        df = df.dropna(subset=["power"]).drop(columns="cum_kwh").reset_index(drop=True)

        if not df.empty:
            power_min = float(df["power"].min())
            power_max = float(df["power"].max())
            global_min = min(global_min, power_min)
            global_max = max(global_max, power_max)

        df = assign_buckets(df, inplace=True)
        df["source"] = path.stem

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    if math.isfinite(global_min) and math.isfinite(global_max):
        result.attrs["power_stats"] = {
            "min": global_min,
            "max": global_max,
            "unit": "kW",
        }
    result.attrs["dropped_negative_deltas"] = dropped_negative_deltas

    # Normalize over the concatenated data so the scale matches the global
    # min/max exported to PSDM (per-file scaling would mix incompatible scales).
    if normalize:
        result = normalize_power(result, col="power", eps=eps)

    if discretize:
        result = discretize_power(result, col="power", state_col="state")

    return result
