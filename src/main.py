from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.markov.plothelper import plot_transition_matrix
from src.markov.transmats import build_transition_matrices_parallel
from src.preprocessing.bucketing import add_bucket_columns
from src.preprocessing.loader import load_timeseries

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def plot_state_distribution(df):

    counts = df["state"].value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xlabel("State")
    plt.ylabel("Anzahl Einträge")
    plt.title("Verteilung der Einträge nach State")
    plt.xticks(counts.index)
    plt.show()


def main():
    df = load_timeseries(normalize=True, discretize=True)
    df = add_bucket_columns(df)

    n_states = int(df["state"].max()) + 1

    counts, probs = build_transition_matrices_parallel(
        df,
        n_states=n_states,
        alpha=0.5,
        n_jobs=-1,
    )

    out_file = PROCESSED_DIR / "transition_matrices.npz"
    np.savez_compressed(out_file, counts=counts, probs=probs)
    print(
        f"Gespeichert unter {out_file} "
        f"(counts: {counts.shape}, probs: {probs.shape})"
    )

    data = np.load(PROCESSED_DIR / "transition_matrices.npz")
    probs = data["probs"]

    bucket_ids_to_plot = [0, 1767]

    for bid in bucket_ids_to_plot:
        plot_transition_matrix(probs[bid], bid)


if __name__ == "__main__":
    main()
