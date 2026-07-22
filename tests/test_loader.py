from pathlib import Path

import pandas as pd
import pytest

import src.preprocessing.loader as loader_module
from src.markov.buckets import bucket_id
from src.preprocessing.loader import load_timeseries


def _configure_input(monkeypatch, **overrides) -> None:
    config = {
        "skiprows": 21,
        "timestamp_col": "Zeitstempel",
        "value_col": "Messwert",
        "value_representation": "cumulative_energy",
        "interval_minutes": 15,
        "drop_negative_deltas": True,
    }
    config.update(overrides)
    monkeypatch.setitem(loader_module.CONFIG, "input", config)


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

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch)

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


def _write_csv(path: Path, cum_kwh: list[float | str]) -> None:
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

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch)

    df = load_timeseries(normalize=True, discretize=True)

    assert df.attrs["power_stats"] == {"min": 1.0, "max": 6.0, "unit": "kW"}

    # Normalized against global [1.0, 6.0], not per file: the 2.0 kW value
    # maps to 0.2 in both files instead of 1.0 in file A and 0.0 in file B.
    expected_power = pd.Series([0.0, 0.2, 0.2, 1.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)


def test_load_timeseries_uses_explicit_data_dir(tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    _write_csv(first_dir / "first_source.csv", [0.0, 0.25])
    _write_csv(second_dir / "second_source.csv", [0.0, 0.5])
    _configure_input(monkeypatch)

    first = load_timeseries(data_dir=first_dir)
    second = load_timeseries(data_dir=second_dir)

    assert set(first["source"]) == {"first_source"}
    assert set(second["source"]) == {"second_source"}


def test_load_timeseries_drops_negative_deltas(tmp_path, monkeypatch):
    """Negative cumulative meter deltas are treated as invalid reset/correction rows."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    _write_csv(raw_dir / "reset.csv", [1.0, 1.5, 1.0, 1.25])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch)

    df = load_timeseries(normalize=False, discretize=False)

    expected_power = pd.Series([2.0, 1.0], name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected_power)
    assert df.attrs["power_stats"] == {"min": 1.0, "max": 2.0, "unit": "kW"}
    assert df.attrs["dropped_negative_deltas"] == 1


@pytest.mark.parametrize(
    ("representation", "values", "expected_power"),
    [
        ("interval_energy", [0.25, -0.5, 1.0], [1.0, -2.0, 4.0]),
        ("power", [0.25, -0.5, 1.0], [0.25, -0.5, 1.0]),
    ],
)
def test_load_timeseries_value_representations(
    tmp_path, monkeypatch, representation, values, expected_power
):
    """Interval energy is converted to kW while power is used directly."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", values)

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch, value_representation=representation)

    df = load_timeseries(normalize=False, discretize=False)

    expected = pd.Series(expected_power, name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected)
    assert len(df) == len(values)
    assert df.attrs["dropped_negative_deltas"] == 0
    assert df.attrs["dropped_missing_values"] == 0


@pytest.mark.parametrize(
    ("representation", "expected_power"),
    [
        ("interval_energy", [1.0, 4.0]),
        ("power", [0.25, 1.0]),
    ],
)
def test_load_timeseries_counts_missing_input_values(
    tmp_path, monkeypatch, representation, expected_power
):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [0.25, "", 1.0])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch, value_representation=representation)

    df = load_timeseries(normalize=False, discretize=False)

    expected = pd.Series(expected_power, name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected)
    assert len(df) == 2
    assert df.attrs["dropped_missing_values"] == 1


@pytest.mark.parametrize("representation", ["cumulative_energy", "interval_energy"])
def test_energy_input_requires_interval_minutes(tmp_path, monkeypatch, representation):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [0.0, 0.25])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    monkeypatch.setitem(
        loader_module.CONFIG,
        "input",
        {
            "skiprows": 21,
            "timestamp_col": "Zeitstempel",
            "value_col": "Messwert",
            "value_representation": representation,
        },
    )

    with pytest.raises(ValueError, match="interval_minutes must be a positive"):
        load_timeseries()


@pytest.mark.parametrize(
    ("representation", "interval_minutes"),
    [
        ("cumulative_energy", 0),
        ("interval_energy", -15),
        ("cumulative_energy", True),
        ("interval_energy", "15"),
        ("power", 0),
        ("power", float("inf")),
    ],
)
def test_interval_minutes_must_be_positive_when_configured(
    tmp_path, monkeypatch, representation, interval_minutes
):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [0.0, 0.25])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(
        monkeypatch,
        value_representation=representation,
        interval_minutes=interval_minutes,
    )

    with pytest.raises(ValueError, match="interval_minutes must be a positive"):
        load_timeseries()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"value_representation": "voltage"}, "value_representation must be one"),
        ({"factor": 4.0}, "input.factor is no longer supported"),
        ({"pools": "true"}, "input.pools must be a boolean"),
    ],
)
def test_load_timeseries_rejects_invalid_or_legacy_config(
    tmp_path, monkeypatch, overrides, message
):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [0.0, 0.25])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(monkeypatch, **overrides)

    with pytest.raises(ValueError, match=message):
        load_timeseries()


@pytest.mark.parametrize(
    "representation", ["cumulative_energy", "interval_energy", "power"]
)
def test_drop_negative_deltas_must_be_boolean(tmp_path, monkeypatch, representation):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [1.0, -1.0])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    _configure_input(
        monkeypatch,
        value_representation=representation,
        drop_negative_deltas="true",
    )

    with pytest.raises(ValueError, match="drop_negative_deltas must be a boolean"):
        load_timeseries()


@pytest.mark.parametrize(
    ("drop_negative_deltas", "expected_power", "dropped_count"),
    [
        (None, [2.0, 1.0], 1),
        (False, [2.0, -2.0, 1.0], 0),
    ],
)
def test_cumulative_negative_delta_policy(
    tmp_path,
    monkeypatch,
    drop_negative_deltas,
    expected_power,
    dropped_count,
):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample.csv", [1.0, 1.5, 1.0, 1.25])

    monkeypatch.setattr(loader_module, "DATA_DIR", raw_dir)
    config = {
        "skiprows": 21,
        "timestamp_col": "Zeitstempel",
        "value_col": "Messwert",
        "value_representation": "cumulative_energy",
        "interval_minutes": 15,
    }
    if drop_negative_deltas is not None:
        config["drop_negative_deltas"] = drop_negative_deltas
    monkeypatch.setitem(loader_module.CONFIG, "input", config)

    df = load_timeseries(normalize=False, discretize=False)

    expected = pd.Series(expected_power, name="power", dtype="float32")
    pd.testing.assert_series_equal(df["power"], expected)
    assert df.attrs["dropped_negative_deltas"] == dropped_count


if __name__ == "__main__":
    pytest.main([__file__])
