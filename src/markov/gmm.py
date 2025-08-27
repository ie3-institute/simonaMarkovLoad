import math

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .buckets import NUM_BUCKETS
from .transition_counts import N_STATES

__all__ = ["GaussianBucketModels", "fit_gmms", "sample_value"]

GmmTuple = tuple[np.ndarray, np.ndarray, np.ndarray]
GaussianBucketModels = list[list[GmmTuple | None]]

_rng = np.random.default_rng()


def _fit_single(
    x: np.ndarray,
    *,
    min_samples: int = 30,
    k_candidates: tuple[int, ...] = (1, 2, 3),
    random_state: int | None = None,
) -> GmmTuple | None:
    if x.size == 0:
        return None
    if len(x) < min_samples:
        mean = float(np.mean(x))
        var = float(max(np.var(x), 1e-5))
        return (
            np.array([1.0], dtype=float),
            np.array([mean], dtype=float),
            np.array([var], dtype=float),
        )
    best_bic = math.inf
    best_gmm: GaussianMixture | None = None
    for k in k_candidates:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="spherical",
            n_init=1,
            random_state=random_state,
        ).fit(x.reshape(-1, 1))
        bic = gmm.bic(x.reshape(-1, 1))
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    return (
        best_gmm.weights_,
        best_gmm.means_.ravel(),
        best_gmm.covariances_.ravel(),
    )


def fit_gmms(
    df: pd.DataFrame,
    *,
    value_col: str = "x",
    bucket_col: str = "bucket",
    state_col: str = "state",
    min_samples: int = 5,
    k_candidates: tuple[int, ...] = (1, 2, 3),
    n_jobs: int = -1,
    random_state: int | None = None,
    verbose: int = 0,
    heartbeat_seconds: int | None = None,
) -> GaussianBucketModels:
    if heartbeat_seconds:
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(heartbeat_seconds, repeat=True)

    grouped = (
        df[[bucket_col, state_col, value_col]]
        .groupby([bucket_col, state_col])[value_col]
        .apply(list)
        .to_dict()
    )

    tasks = [
        ((b, s), grouped.get((b, s), []))
        for b in range(NUM_BUCKETS)
        for s in range(N_STATES)
    ]

    iterable = (
        tqdm(tasks, desc="Fitting GMMs", unit="model") if verbose and tqdm else tasks
    )

    def _worker(item):
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


def sample_value(
    gmms: GaussianBucketModels,
    bucket: int,
    state: int,
    rng: np.random.Generator | None = None,
) -> float:
    gmm = gmms[bucket][state]
    if gmm is None:
        raise ValueError(f"No GMM trained for bucket {bucket}, state {state}")
    weights, means, vars_ = gmm
    rng = _rng if rng is None else rng
    comp = rng.choice(len(weights), p=weights)
    # Clamp to [0, 1] since values are normalized
    val = float(rng.normal(means[comp], math.sqrt(vars_[comp])))
    return float(min(1.0, max(0.0, val)))
