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
    """Loader must return timestamp, power, scaled, state, bucket."""
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

    # two rows (first diff removed) & five columns
    assert df.shape == (2, 5)
    assert list(df.columns) == ["timestamp", "power", "scaled", "state", "bucket"]

    # timestamps
    expected_ts = pd.Series(
        pd.to_datetime(["2021-01-01 00:15:00", "2021-01-01 00:30:00"]),
        name="timestamp",
    )
    pd.testing.assert_series_equal(df["timestamp"], expected_ts, check_names=False)

    # power column ( diff * 4 )
    expected_power = pd.Series([2.0, 2.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)

    # scaled column – both values identical → expect 0 with the given eps logic
    expected_scaled = pd.Series([0.0, 0.0], name="scaled", dtype="float32")
    pd.testing.assert_series_equal(df["scaled"], expected_scaled)

    # state column – with scaled=0, discretiser puts them in bin 0
    expected_state = pd.Series([0, 0], name="state", dtype="uint8")
    pd.testing.assert_series_equal(df["state"], expected_state)

    expected_bucket = pd.Series(
        [bucket_id(ts) for ts in expected_ts], name="bucket", dtype="uint16"
    )
    pd.testing.assert_series_equal(df["bucket"], expected_bucket)


if __name__ == "__main__":
    pytest.main([__file__])
