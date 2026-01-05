from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.config import CONFIG
from src.export import export_psdm_json
from src.markov.buckets import assign_buckets, bucket_id
from src.markov.gmm import fit_gmms, sample_value
from src.markov.transition_counts import build_transition_counts
from src.markov.transitions import build_transition_matrices
from src.preprocessing.loader import load_timeseries

SIM_DAYS = 10
PER_DAY = 96


def _detect_value_col(df: pd.DataFrame) -> str:
    for c in ["x", "value", "load", "power", "p_norm", "load_norm"]:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    raise KeyError("numeric load column missing")


def simulate_step(
    probs: np.ndarray, gmms, bucket: int, state: int, rng: np.random.Generator
) -> tuple[int, float]:
    """Robust simulation step that handles missing GMMs."""

    transitions = probs[bucket, state, :].copy()

    valid_states = np.array(
        [gmms[bucket][s] is not None for s in range(len(transitions))]
    )

    if not np.any(valid_states):
        return state, 0.0

    transitions[~valid_states] = 0.0

    if transitions.sum() > 0:
        transitions = transitions / transitions.sum()
    else:

        return state, 0.0

    next_state = rng.choice(len(transitions), p=transitions)

    sampled_value = sample_value(gmms, bucket, next_state, rng=rng)

    return next_state, sampled_value


def _simulate_series(
    probs: np.ndarray,
    gmms,
    start_ts: pd.Timestamp,
    start_state: int,
    periods: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng() if rng is None else rng
    ts = pd.date_range(start_ts, periods=periods, freq="15min")
    states = np.empty(periods, dtype=int)
    xs = np.empty(periods, dtype=float)

    s = start_state
    for i, t in enumerate(ts):
        b = bucket_id(t)
        s, x = simulate_step(probs, gmms, b, s, rng)
        states[i] = s
        xs[i] = x

    return pd.DataFrame({"timestamp": ts, "state": states, "x_sim": xs})


def _plot_first_25_buckets(counts: np.ndarray, probs: np.ndarray) -> None:
    buckets = range(25)
    fig, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=True, sharey=True)
    vmax = probs[list(buckets)].max()
    norm = Normalize(vmin=0, vmax=vmax)

    for idx, b in enumerate(buckets):
        ax = axes.flat[idx]
        if counts[b].sum() == 0:
            ax.axis("off")
            continue
        im = ax.imshow(probs[b], aspect="auto", origin="lower", norm=norm)
        ax.set_title(f"Bucket {b}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(buckets) :]:
        ax.axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="p")
    fig.suptitle("Transition probabilities – buckets 0–24", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.97, 0.96])
    plt.show()


def _plot_simulation_diagnostics(
    df: pd.DataFrame, sim: pd.DataFrame, value_col: str
) -> None:
    first_day = sim.iloc[:PER_DAY]
    plt.figure(figsize=(10, 3))
    plt.plot(first_day["timestamp"], first_day["x_sim"], marker=".")
    plt.title("Simulated power – first day")
    plt.ylabel("normalised load x")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(df[value_col], bins=50, alpha=0.6, density=True, label="original")
    plt.hist(sim["x_sim"], bins=50, alpha=0.6, density=True, label="simulated")
    plt.title("Original vs simulated load distribution")
    plt.xlabel("normalised load x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    sim["hour"] = sim["timestamp"].dt.hour
    plt.figure(figsize=(10, 4))
    sim.boxplot(column="x_sim", by="hour", grid=False)
    plt.title("Simulated power by hour of day")
    plt.xlabel("hour of day")
    plt.ylabel("normalised load x")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_timeseries(normalize=True, discretize=True)
    power_stats = df.attrs.get("power_stats", {})
    reference_power = power_stats.get("max")
    min_power = power_stats.get("min")
    if "bucket" not in df.columns:
        df = assign_buckets(df)

    val_col = _detect_value_col(df)

    counts = build_transition_counts(df)
    probs = build_transition_matrices(df)

    _plot_first_25_buckets(counts, probs)

    gm_kwargs = {
        "value_col": val_col,
        "verbose": 1,
        "heartbeat_seconds": 60,
    }

    gmms = fit_gmms(df, **gm_kwargs)

    meta = {"records": len(df)}
    if "ts" in df.columns:
        meta["time_range"] = {"start": str(df["ts"].min()), "end": str(df["ts"].max())}
    if "source" in CONFIG.get("data", {}):
        meta["source"] = CONFIG["data"]["source"]

    try:
        out_path = Path(
            CONFIG.get("output", {}).get("psdm_json", "out/psdm_model.json")
        )
        pretty = bool(CONFIG.get("output", {}).get("pretty_json", False))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_psdm_json(
            out_path,
            df,
            probs,
            gmms,
            meta=meta,
            gmm_params=gm_kwargs,
            pretty=pretty,
            reference_power_kw=reference_power,
            min_power_kw=min_power,
        )
        print(f"[export] PSDM JSON written to {out_path}")
    except Exception as e:
        print(f"[export] FAILED to write PSDM JSON: {e}")

    periods = SIM_DAYS * PER_DAY
    sim = _simulate_series(
        probs,
        gmms,
        start_ts=df["timestamp"].min().normalize(),
        start_state=int(df["state"].iloc[0]),
        periods=periods,
    )

    _plot_simulation_diagnostics(df, sim, val_col)


if __name__ == "__main__":
    main()
