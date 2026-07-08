from pathlib import Path

import pandas as pd
import pytest

import src.preprocessing.loader as loader_module
from src.markov.buckets import bucket_id
from src.preprocessing.loader import load_timeseries


def _create_sample_csv(path: Path) -> None:
    """Write 3 kWh readings; first diff will be NaN and dropped."""
    with path.open("w", encoding="utf-8") as f:
        for _ in range(21):
            f.write("metadata line\n")
        f.write("Zeitstempel,Messwert\n")
        f.write("2021-01-01 00:00:00,0.0\n")
        f.write("2021-01-01 00:15:00,0.5\n")
        f.write("2021-01-01 00:30:00,1.0\n")


def test_load_timeseries_full(tmp_path, monkeypatch):
    """Loader must return timestamp, power state, bucket."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_file = raw_dir / "sample.csv"
    _create_sample_csv(csv_file)

    monkeypatch.setattr(loader_module, "RAW_DATA_DIR", raw_dir)
    loader_module.CONFIG["input"].update(
        {
            "timestamp_col": "Zeitstempel",
            "value_col": "Messwert",
            "factor": 4.0,
        }
    )

    df = load_timeseries(normalize=True, discretize=True)

    assert df.shape == (2, 5)
    assert list(df.columns) == ["timestamp", "power", "bucket", "source", "state"]

    expected_ts = pd.Series(
        pd.to_datetime(["2021-01-01 00:15:00", "2021-01-01 00:30:00"]),
        name="timestamp",
    )
    pd.testing.assert_series_equal(df["timestamp"], expected_ts, check_names=False)

    expected_power = pd.Series([0.0, 0.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)

    expected_state = pd.Series([0, 0], name="state", dtype="int64")
    pd.testing.assert_series_equal(df["state"], expected_state)

    expected_bucket = pd.Series(
        [bucket_id(ts) for ts in expected_ts], name="bucket", dtype="uint16"
    )
    pd.testing.assert_series_equal(df["bucket"], expected_bucket)

    stats = df.attrs.get("power_stats")
    assert stats == {"min": 2.0, "max": 2.0, "unit": "kW"}


def _write_csv(path: Path, cum_kwh: list[float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _ in range(21):
            f.write("metadata line\n")
        f.write("Zeitstempel,Messwert\n")
        for i, value in enumerate(cum_kwh):
            f.write(f"2021-01-01 00:{i * 15:02d}:00,{value}\n")


def test_load_timeseries_normalizes_globally(tmp_path, monkeypatch):
    """Files with different power ranges must share one global scale."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # File A: powers 1.0 and 2.0 kW; file B: powers 2.0 and 6.0 kW.
    _write_csv(raw_dir / "a.csv", [0.0, 0.25, 0.75])
    _write_csv(raw_dir / "b.csv", [0.0, 0.5, 2.0])

    monkeypatch.setattr(loader_module, "RAW_DATA_DIR", raw_dir)
    loader_module.CONFIG["input"].update(
        {
            "timestamp_col": "Zeitstempel",
            "value_col": "Messwert",
            "factor": 4.0,
        }
    )

    df = load_timeseries(normalize=True, discretize=True)

    assert df.attrs["power_stats"] == {"min": 1.0, "max": 6.0, "unit": "kW"}

    # Normalized against global [1.0, 6.0], not per file: the 2.0 kW value
    # maps to 0.2 in both files instead of 1.0 in file A and 0.0 in file B.
    expected_power = pd.Series([0.0, 0.2, 0.2, 1.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)


def test_load_timeseries_drops_negative_deltas(tmp_path, monkeypatch):
    """Negative cumulative meter deltas are treated as invalid reset/correction rows."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    _write_csv(raw_dir / "reset.csv", [1.0, 1.5, 1.0, 1.25])

    monkeypatch.setattr(loader_module, "RAW_DATA_DIR", raw_dir)
    loader_module.CONFIG["input"].update(
        {
            "timestamp_col": "Zeitstempel",
            "value_col": "Messwert",
            "factor": 4.0,
            "drop_negative_deltas": True,
        }
    )

    df = load_timeseries(normalize=False, discretize=False)

    expected_power = pd.Series([2.0, 1.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)
    assert df.attrs["power_stats"] == {"min": 1.0, "max": 2.0, "unit": "kW"}
    assert df.attrs["dropped_negative_deltas"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
