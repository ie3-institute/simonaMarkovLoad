import numpy as np
import pandas as pd

from src.config import CONFIG

from ._core import _transition_counts

alpha = float(CONFIG["model"]["laplace_alpha"])


def build_transition_matrices(df: pd.DataFrame, *, dtype=np.float32) -> np.ndarray:
    counts = _transition_counts(df, dtype=dtype)
    counts += alpha
    counts /= counts.sum(axis=2, keepdims=True)
    return counts.astype(dtype)
