"""Laplace-smoothed transition probability matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import _transition_counts


def build_transition_matrices(
    df: pd.DataFrame, *, alpha: float = 1.0, dtype=np.float32
) -> np.ndarray:
    counts = _transition_counts(df, dtype=dtype)
    counts += alpha
    counts /= counts.sum(axis=2, keepdims=True)
    return counts.astype(dtype)
