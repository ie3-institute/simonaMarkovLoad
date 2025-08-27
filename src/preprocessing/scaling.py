import numpy as np
import pandas as pd


def normalize_power(
    df: pd.DataFrame,
    *,
    col: str = "power",
    eps: float = 1e-12,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty, nothing to normalize.")

    p_min = df[col].min()
    p_max = df[col].max()
    denom = (p_max - p_min) or eps  # anti zero division

    df[col] = (df[col] - p_min) / denom
    return df


def discretize_power(
    df: pd.DataFrame,
    *,
    col: str = "power",
    state_col: str = "state",
) -> pd.DataFrame:
    taus = np.array([(k / 10) ** 2 for k in range(1, 10)], dtype=float)

    values = df[col].to_numpy()
    states = np.searchsorted(taus, values, side="right")

    df[state_col] = states
    return df


def discretize_states(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    state_col: str = "state",
) -> pd.DataFrame:
    """Discretize continuous values into discrete states using thresholds.

    This is a wrapper around discretize_power with more generic naming.
    """
    return discretize_power(df, col=value_col, state_col=state_col)
