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
    denom = (p_max - p_min) or eps  # schützt vor ZeroDivisionError

    df[col] = (df[col] - p_min) / denom
    return df
