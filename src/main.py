import matplotlib.pyplot as plt

from src.preprocessing.loader import load_timeseries


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
    df = load_timeseries()
    print(df)
    df_norm = load_timeseries(normalize=True)
    print(df_norm)
    df_disc = load_timeseries(normalize=True, discretize=True)
    print(df_disc)
    plot_state_distribution(df_disc)


if __name__ == "__main__":
    main()
