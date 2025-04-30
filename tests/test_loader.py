from pathlib import Path

import pandas as pd
import pytest

import src.preprocessing.loader as loader_module

# Import the module under test
from src.preprocessing.loader import load_timeseries


def create_sample_csv(path: Path):
    # Write 21 dummy metadata lines
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(21):
            f.write("metadata line\n")
        # Header and sample data
        f.write("Zeitstempel,Messwert\n")
        f.write("2021-01-01 00:00:00,0.0\n")
        f.write("2021-01-01 00:15:00,0.5\n")
        f.write("2021-01-01 00:30:00,1.0\n")


def test_load_timeseries(tmp_path, monkeypatch):
    # Setup a temporary data/raw directory
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_file = raw_dir / "sample.csv"
    create_sample_csv(csv_file)

    # Monkeypatch the RAW_DATA_DIR in the loader module
    monkeypatch.setattr(loader_module, "RAW_DATA_DIR", raw_dir)

    # Execute loader (skip normalize and discretize)
    df = load_timeseries()

    # Expect two rows (first diff yields NaN and is dropped)
    assert df.shape == (2, 2)
    # Columns should be timestamp and power
    assert list(df.columns) == ["timestamp", "power"]

    # Check timestamps are parsed correctly
    expected_timestamps = pd.Series(
        pd.to_datetime(["2021-01-01 00:15:00", "2021-01-01 00:30:00"]), name="timestamp"
    )
    pd.testing.assert_series_equal(
        df["timestamp"], expected_timestamps, check_names=False
    )

    # Check power calculation: diff * 4
    # Values: (0.5 - 0.0)*4 = 2.0, (1.0 - 0.5)*4 = 2.0
    expected_power = pd.Series([2.0, 2.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)


if __name__ == "__main__":
    pytest.main([__file__])
