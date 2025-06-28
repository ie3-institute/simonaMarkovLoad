import math
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    # rich progress bar for terminals / notebooks; optional dependency
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None  # type: ignore

from .buckets import NUM_BUCKETS
from .transition_counts import N_STATES

__all__ = [
    "GaussianBucketModels",
    "fit_gmms",
    "sample_value",
]


GmmTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
GaussianBucketModels = List[List[GmmTuple]]


def _fit_single(
    x: np.ndarray,
    *,
    min_samples: int = 30,
    k_candidates: Tuple[int, ...] = (1, 2, 3),
    random_state: int | None = None,
) -> GmmTuple:
    """Fit 1‑3 component spherical GMM; fallback to Normal if too few samples."""
    if len(x) < min_samples:
        mean = float(np.mean(x))
        var = float(np.var(x) + 1e-6)
        return (np.array([1.0]), np.array([mean]), np.array([var]))

    best_bic = math.inf
    best_gmm: GaussianMixture | None = None
    for k in k_candidates:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="spherical",
            n_init="auto",
            random_state=random_state,
        ).fit(x.reshape(-1, 1))
        bic = gmm.bic(x.reshape(-1, 1))
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    weights = best_gmm.weights_
    means = best_gmm.means_.ravel()
    variances = best_gmm.covariances_.ravel()
    return (weights, means, variances)


def fit_gmms(
    df: pd.DataFrame,
    *,
    value_col: str = "x",
    bucket_col: str = "bucket",
    state_col: str = "state",
    min_samples: int = 30,
    k_candidates: Tuple[int, ...] = (1, 2, 3),
    n_jobs: int = -1,
    random_state: int | None = None,
    verbose: int = 0,
) -> GaussianBucketModels:
    """Return list [bucket][state] -> (weights, means, variances)."""

    samples: Dict[Tuple[int, int], List[float]] = {}
    for _, row in df[[bucket_col, state_col, value_col]].iterrows():
        samples.setdefault((row[bucket_col], row[state_col]), []).append(row[value_col])

    tasks = [
        ((b, s), samples.get((b, s), []))
        for b in range(NUM_BUCKETS)
        for s in range(N_STATES)
    ]

    iterable = tasks
    if verbose > 0 and tqdm is not None:
        iterable = tqdm(tasks, desc="Fitting GMMs", unit="model")

    def _worker(item: Tuple[Tuple[int, int], List[float]]):
        (b, s), x_list = item
        x = np.asarray(x_list, dtype=float)
        return (
            b,
            s,
            _fit_single(
                x,
                min_samples=min_samples,
                k_candidates=k_candidates,
                random_state=random_state,
            ),
        )

    results = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_worker)(task) for task in iterable
    )

    gmms: GaussianBucketModels = [
        [None for _ in range(N_STATES)] for _ in range(NUM_BUCKETS)
    ]
    for b, s, gmm_tuple in results:
        gmms[b][s] = gmm_tuple
    return gmms


_rng = np.random.default_rng()


def sample_value(
    gmms: GaussianBucketModels,
    bucket: int,
    state: int,
    rng: np.random.Generator | None = None,
) -> float:
    """Draw a normalised load value from the GMM for (bucket, state)."""
    weights, means, vars_ = gmms[bucket][state]
    rng = _rng if rng is None else rng
    comp = rng.choice(len(weights), p=weights)
    return float(rng.normal(means[comp], math.sqrt(vars_[comp])))
