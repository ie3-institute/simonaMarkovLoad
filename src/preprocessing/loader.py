import math
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from .constant_loads import load_constant_loads

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
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    cfg_in = CONFIG["input"]
    cfg_splitting = CONFIG["splitting"]
    representation = _validate_input_config(cfg_in)

    input_dir = DATA_DIR if data_dir is None else data_dir

    # Collect all .csv files in the input directory
    csv_files: list[Path] = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}.")

    path_to_frame: dict[str, pd.DataFrame] = {}

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

        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M")
        _convert_to_power(df, cfg_in, representation)
        path_to_frame[path.stem] = df

    if cfg_splitting is not None:
        value_column = "power"

        rm = cfg_splitting.get("remove_constants", False)
        if rm:
            constants = load_constant_loads()
        else:
            constants = None


        frames: dict[str, dict[str, pd.DataFrame]] = {}
        frames["base"] = {}
        frames["variation"] = {}

        for path, df in path_to_frame.items():
            if constants is not None and path in constants:
                current_constants = constants[path]

                # remove constant loads
                if current_constants is not None:
                    for c in current_constants:
                        (threshold, start_hour, end_hour) = c
                        mask = ((df.index.hour >= start_hour) | (df.index.hour < end_hour)) & (df[value_column] > threshold)

                        const_df = _copy_and_drop(df, False)

                        const_df[value_column] = 0.0
                        const_df.loc[mask, value_column] = threshold

                        df[value_column] = df[value_column] - const_df[value_column]

                        name = f"{threshold}_{start_hour}_{end_hour}"
                        if name not in frames:
                            frames[name] = {}

                        frames[name][path] = const_df

            if cfg_splitting["value_representation"] == "absolute":
                sm = cfg_splitting["value"]
            else:
                sm = df.nsmallest(int(len(df) * cfg_splitting["value"]), value_column)[value_column].max()

            base_df = _copy_and_drop(df, False)
            value_mask = base_df[value_column] > sm
            base_df[value_column] = base_df[value_column].mask(value_mask)
            base_df[value_column] = base_df[value_column].fillna(sm)

            variation_df = _copy_and_drop(df)
            variation_df[value_column] = df[value_column] - base_df[value_column]

            frames["base"][path] = base_df
            frames["variation"][path] = variation_df

        res: dict[str, pd.DataFrame] = {}

        for name, values in frames.items():
            res[name] = _process_timeseries(values, cfg_in, representation, normalize, discretize, eps)

        return res

    else:
        return _process_timeseries(
            path_to_frame,
            cfg_in,
            representation,
            normalize,
            discretize,
            eps
        )


def _process_timeseries(
    dfs: dict[str, pd.DataFrame],
    cfg_in: dict,
    representation: str,
    normalize: bool = False,
    discretize: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    # Read columns (ts, value) from each file
    frames: list[pd.DataFrame] = []
    global_min = math.inf
    global_max = -math.inf
    dropped_negative_deltas = 0
    dropped_missing_values = 0

    for path, df in dfs.items():
        dropped_missing_values += int(df["input_value"].isna().sum())

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
        df["source"] = path

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


def _copy_and_drop(df: pd.DataFrame, drop: bool=True) -> pd.DataFrame:
    tmp = df.copy(deep=True)

    if drop:
        return tmp.drop(columns=["power"])
    else:
        return tmp


