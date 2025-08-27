import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from src.markov.buckets       import assign_buckets, bucket_id
from src.markov.gmm           import fit_gmms, sample_value
from src.markov.transition_counts import build_transition_counts
from src.markov.transitions   import build_transition_matrices
from src.markov.postprocess   import apply_dwell_time, two_point_smooth
from src.preprocessing.loader import load_timeseries

SIM_DAYS = 10
PER_DAY  = 96

def _detect_value_col(df: pd.DataFrame) -> str:
    for c in ["x", "value", "load", "power", "p_norm", "load_norm"]:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    raise KeyError("numeric load column missing")


def _simulate_states(
    probs: np.ndarray,
    start_ts: pd.Timestamp,
    start_state: int,
    periods: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng

    states = np.empty(periods, dtype=int)
    s = start_state
    for i, t in enumerate(pd.date_range(start_ts, periods=periods, freq="15min")):
        b  = bucket_id(t)
        s  = rng.choice(probs.shape[1], p=probs[b, s])
        states[i] = s
    return states


def _states_to_series(
    states: np.ndarray,
    gmms,
    start_ts: pd.Timestamp,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    rng = np.random.default_rng() if rng is None else rng
    ts  = pd.date_range(start_ts, periods=len(states), freq="15min")
    xs  = np.empty_like(states, dtype=float)
    for i, (t, s) in enumerate(zip(ts, states)):
        b   = bucket_id(t)
        xs[i] = sample_value(gmms, b, s, rng=rng)
    return pd.Series(xs, index=ts, name="x_sim")



def _plot_simulation_diagnostics(df: pd.DataFrame,
                                 sim: pd.Series,
                                 value_col: str) -> None:
    first_day = sim.iloc[:PER_DAY]
    plt.figure(figsize=(10, 3))
    plt.plot(first_day.index, first_day, marker=".")
    plt.title("Simulated power – first day")
    plt.ylabel("normalised load x")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(df[value_col], bins=50, alpha=0.6, density=True, label="original")
    plt.hist(sim,           bins=50, alpha=0.6, density=True, label="simulated")
    plt.title("Original vs simulated load distribution")
    plt.xlabel("normalised load x")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    sim_hr = pd.DataFrame({"x_sim": sim, "hour": sim.index.hour})
    plt.figure(figsize=(10, 4))
    sim_hr.boxplot(column="x_sim", by="hour", grid=False)
    plt.title("Simulated power by hour of day")
    plt.xlabel("hour of day")
    plt.ylabel("normalised load x")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_timeseries(normalize=True, discretize=True)
    if "bucket" not in df.columns:
        df = assign_buckets(df)

    val_col = _detect_value_col(df)

    counts = build_transition_counts(df)
    probs  = build_transition_matrices(df)

    gmms   = fit_gmms(df, value_col=val_col, verbose=1,
                      heartbeat_seconds=60)

    periods  = SIM_DAYS * PER_DAY
    start_ts = df["timestamp"].min().normalize()
    raw_states = _simulate_states(
        probs,
        start_ts=start_ts,
        start_state=int(df["state"].iloc[0]),
        periods=periods,
    )

    states = apply_dwell_time(raw_states, p_extend=0.7)

    sim_series = _states_to_series(states, gmms, start_ts)
    sim_series = two_point_smooth(sim_series)

    real_kwh = df.set_index("timestamp")[val_col].resample("D").sum() / 4
    sim_kwh  = sim_series.resample("D").sum() / 4
    scale = df[val_col].mean() / sim_series.mean()
    sim_series *= scale

    _plot_simulation_diagnostics(df, sim_series, val_col)


if __name__ == "__main__":
    main()
