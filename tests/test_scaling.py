import numpy as np
import pandas as pd
import pytest

from src.preprocessing.scaling import discretize_power, normalize_power

# --------------------------------------------------------------------------- #
# normalize_power                                                             #
# --------------------------------------------------------------------------- #


def test_normalize_power_basic():
    df = pd.DataFrame({"power": [0.0, 2.0, 4.0]})
    out = normalize_power(df.copy())

    expected = pd.Series([0.0, 0.5, 1.0], name="power")
    pd.testing.assert_series_equal(out["power"], expected, check_dtype=False)


def test_normalize_power_constant_values():
    df = pd.DataFrame({"power": [3.3, 3.3, 3.3]})
    out = normalize_power(df.copy(), eps=1e-12)

    assert out["power"].eq(0.0).all()


def test_normalize_power_empty_df_raises():
    empty = pd.DataFrame(columns=["power"])
    with pytest.raises(ValueError):
        normalize_power(empty)


# --------------------------------------------------------------------------- #
# discretize_power                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected_state",
    [
        (0.00, 0),
        (0.02, 1),
        (0.05, 2),
        (0.20, 4),
        (0.83, 9),
    ],
)
def test_discretize_power_states(value, expected_state):
    df = pd.DataFrame({"power": [value]})
    out = discretize_power(df.copy())

    assert out.loc[0, "state"] == expected_state


def test_discretize_power_preserves_power_column():
    values = np.linspace(0, 1, 6)
    df = pd.DataFrame({"power": values})
    out = discretize_power(df.copy())

    pd.testing.assert_series_equal(out["power"], pd.Series(values, name="power"))
