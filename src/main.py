import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.markov.buckets import assign_buckets, bucket_id
from src.markov.gmm import fit_gmms, sample_value
from src.markov.transition_counts import build_transition_counts
from src.markov.transitions import build_transition_matrices
from src.preprocessing.loader import load_timeseries

SIM_DAYS = 3


def _detect_value_col(df: pd.DataFrame) -> str:
    candidate_cols = ["x", "value", "load", "power", "p_norm", "load_norm"]
    for c in candidate_cols:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    raise KeyError("No numeric load column found – please inspect the dataframe.")


def _simulate_series(
    probs: np.ndarray,
    gmms,
    start_ts: pd.Timestamp,
    start_state: int,
    periods: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic 15‑min series (timestamp, state, x)."""
    rng = np.random.default_rng() if rng is None else rng
    timestamps = pd.date_range(start_ts, periods=periods, freq="15min")
    states = np.empty(periods, dtype=int)
    xs = np.empty(periods, dtype=float)

    s = start_state
    for i, ts in enumerate(timestamps):
        b = bucket_id(ts)
        s = rng.choice(probs.shape[1], p=probs[b, s])
        states[i] = s
        xs[i] = sample_value(gmms, b, s, rng=rng)
    return pd.DataFrame({"timestamp": timestamps, "state": states, "x_sim": xs})


def main() -> None:
    df = load_timeseries(normalize=True, discretize=True)
    if "bucket" not in df.columns:
        df = assign_buckets(df)

    value_col = _detect_value_col(df)
    print("Using load column:", value_col)

    counts = build_transition_counts(df)
    probs = build_transition_matrices(df)

    _plot_first_25_buckets(counts, probs)

    print("Fitting GMMs … (this may take a moment)")
    gmms = fit_gmms(df, value_col=value_col)

    periods = SIM_DAYS * 96
    sim_df = _simulate_series(
        probs,
        gmms,
        start_ts=df["timestamp"].min().normalize(),
        start_state=int(df["state"].iloc[0]),
        periods=periods,
    )

    _plot_simulation_diagnostics(df, sim_df, value_col)


def _plot_first_25_buckets(counts: np.ndarray, probs: np.ndarray) -> None:
    """Heat‑map grid for buckets 0‑24."""
    buckets = list(range(25))
    fig, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=True, sharey=True)
    vmax = probs[buckets].max()
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
    fig.suptitle("Transition probabilities – buckets 0‑24", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.97, 0.96])
    plt.show()


def _plot_simulation_diagnostics(
    df: pd.DataFrame, sim: pd.DataFrame, value_col: str
) -> None:
    first_day = sim.iloc[:96]
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
    plt.suptitle("")
    plt.title("Simulated power by hour of day")
    plt.xlabel("hour of day")
    plt.ylabel("normalised load x")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
