import math
from pathlib import Path

import pandas as pd

from src.config import CONFIG

from ..markov.buckets import assign_buckets
from .scaling import discretize_power, normalize_power

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

VALUE_REPRESENTATIONS = {"cumulative_energy", "interval_energy", "power"}

__all__ = ["load_timeseries", "DATA_DIR"]


def _validate_input_config(cfg_in: dict) -> str:
    if "factor" in cfg_in:
        raise ValueError(
            "input.factor is no longer supported; configure "
            "input.value_representation and input.interval_minutes instead."
        )

    representation = cfg_in.get("value_representation")
    if representation not in VALUE_REPRESENTATIONS:
        allowed = ", ".join(sorted(VALUE_REPRESENTATIONS))
        raise ValueError(
            f"input.value_representation must be one of: {allowed}; "
            f"got {representation!r}."
        )

    interval_minutes = cfg_in.get("interval_minutes")
    if representation in {"cumulative_energy", "interval_energy"}:
        if isinstance(interval_minutes, bool) or not isinstance(
            interval_minutes, int | float
        ):
            raise ValueError(
                "input.interval_minutes must be a positive number for energy "
                f"input ({representation})."
            )
        if not math.isfinite(interval_minutes) or interval_minutes <= 0:
            raise ValueError(
                "input.interval_minutes must be a positive number for energy "
                f"input ({representation})."
            )
    elif "interval_minutes" in cfg_in and (
        isinstance(interval_minutes, bool)
        or not isinstance(interval_minutes, int | float)
        or not math.isfinite(interval_minutes)
        or interval_minutes <= 0
    ):
        raise ValueError(
            "input.interval_minutes must be a positive number when configured."
        )

    if "drop_negative_deltas" in cfg_in and not isinstance(
        cfg_in["drop_negative_deltas"], bool
    ):
        raise ValueError("input.drop_negative_deltas must be a boolean.")

    if "pools" in cfg_in and not isinstance(cfg_in["pools"], bool):
        raise ValueError("input.pools must be a boolean.")

    return representation


def _convert_to_power(df: pd.DataFrame, cfg_in: dict, representation: str) -> None:
    if representation == "power":
        df["power"] = df["input_value"]
        return

    hours_per_interval = cfg_in["interval_minutes"] / 60
    energy = df["input_value"]
    if representation == "cumulative_energy":
        energy = energy.diff()
    df["power"] = energy / hours_per_interval


def load_timeseries(
    *,
    data_dir: Path | None = None,
    value_dtype: str = "float32",
    normalize: bool = False,
    discretize: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    cfg_in = CONFIG["input"]
    representation = _validate_input_config(cfg_in)

    input_dir = DATA_DIR if data_dir is None else data_dir

    # Collect all .csv files in the input directory
    csv_files: list[Path] = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}.")

    # Read columns (ts, value) from each file
    frames: list[pd.DataFrame] = []
    global_min = math.inf
    global_max = -math.inf
    dropped_negative_deltas = 0
    dropped_missing_values = 0
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
                cfg_in["value_col"]: "input_value",
            }
        )
        dropped_missing_values += int(df["input_value"].isna().sum())

        _convert_to_power(df, cfg_in, representation)

        if representation == "cumulative_energy" and cfg_in.get(
            "drop_negative_deltas", True
        ):
            negative_delta_mask = df["power"] < 0
            dropped_negative_deltas += int(negative_delta_mask.sum())
            df = df.loc[~negative_delta_mask]

        df = (
            df.dropna(subset=["power"])
            .drop(columns="input_value")
            .reset_index(drop=True)
        )

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
    result.attrs["dropped_missing_values"] = dropped_missing_values

    # Normalize over the concatenated data so the scale matches the global
    # min/max exported to PSDM (per-file scaling would mix incompatible scales).
    if normalize:
        result = normalize_power(result, col="power", eps=eps)

    if discretize:
        result = discretize_power(result, col="power", state_col="state")

    return result
