"""Export pre-computed simonaMarkovLoad models to JSON format for PSDM."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG


def gmms_to_json(
    gmms: list[list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]],
) -> list[dict]:
    """Convert GMM models to JSON-serializable format.

    Args:
        gmms: List of bucket models, each containing list of state GMMs

    Returns:
        List of bucket dictionaries with state GMM parameters
    """
    buckets = []

    for _bucket_idx, bucket_states in enumerate(gmms):
        states = []

        for _state_idx, gmm in enumerate(bucket_states):
            if gmm is None:
                states.append(None)
            else:
                weights, means, variances = gmm
                weights_normalized = weights / np.sum(weights)
                variances_safe = np.maximum(variances, 0.0)

                states.append(
                    {
                        "weights": weights_normalized.tolist(),
                        "means": means.tolist(),
                        "variances": variances_safe.tolist(),
                    }
                )

        buckets.append({"states": states})

    return buckets


def transitions_to_json(p: np.ndarray) -> list[list[list[float]]]:
    """Convert transition matrices to JSON-serializable format.

    Args:
        p: Transition probability matrices of shape (n_buckets, n_states, n_states)

    Returns:
        Nested list representation of transition matrices
    """

    return p.astype(np.float32).tolist()


def build_psdm_payload_from_models(
    df: pd.DataFrame,
    p: np.ndarray,
    gmms: list[list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]],
    meta: dict | None = None,
    gmm_params: dict | None = None,
) -> dict:
    """Build the complete PSDM JSON payload from pre-computed models.

    Args:
        df: Training dataframe (used for metadata only)
        p: Pre-computed transition probability matrices of shape (n_buckets, n_states, n_states)
        gmms: Pre-computed GMM models as nested list [bucket][state]
        meta: Optional metadata to include under "training_data" key

    Returns:
        Complete PSDM JSON payload dictionary
    """
    n_states = CONFIG["model"]["n_states"]
    laplace_alpha = CONFIG["model"]["laplace_alpha"]

    thresholds = [(k / 10) ** 2 for k in range(1, 10)]

    if gmm_params is None:
        gmm_params = {
            "max_components": 3,
            "min_samples_per_state": 30,
            "covariance_type": "spherical",
            "random_seed": 42,
        }

    payload = {
        "schema": "simonaMarkovLoad:psdm:1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {
            "name": "simonaMarkovLoad",
            "version": "git:unknown",
            "config": {"n_states": n_states, "laplace_alpha": laplace_alpha},
        },
        "time_model": {
            "bucket_count": 2304,
            "bucket_encoding": {
                "formula": "bucket = month*192 + is_weekend*96 + quarter_hour"
            },
            "sampling_interval_minutes": 15,
            "timezone": "Europe/Berlin",
        },
        "value_model": {
            "value_unit": "normalized",
            "normalization": {"method": "minmax_per_series"},
            "discretization": {"states": n_states, "thresholds_right": thresholds},
        },
        "parameters": {
            "transitions": {"empty_row_strategy": "self_loop"},
            "gmm": gmm_params,
        },
        "data": {
            "transitions": {
                "shape": list(p.shape),
                "dtype": "float32",
                "encoding": "nested_lists",
                "values": transitions_to_json(p),
            },
            "gmms": {"buckets": gmms_to_json(gmms)},
        },
    }

    if meta is not None:
        payload["training_data"] = meta

    return payload


def export_psdm_json(
    path: Path,
    df: pd.DataFrame,
    p: np.ndarray,
    gmms: list[list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]],
    meta: dict | None = None,
    gmm_params: dict | None = None,
    pretty: bool = False,
) -> Path:
    """Export pre-computed models to PSDM JSON format.

    Args:
        path: Output file path
        p: Pre-computed transition probability matrices
        gmms: Pre-computed GMM models
        meta: Optional metadata to include under "training_data" key

    Returns:
        Path to the exported JSON file
    """
    payload = build_psdm_payload_from_models(df, p, gmms, meta, gmm_params)

    indent = 2 if pretty else None

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)

    return path
