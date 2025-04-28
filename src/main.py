import matplotlib.pyplot as plt

from src.preprocessing.loader import load_raw_timeseries
from src.preprocessing.scaling import discretize_power, normalize_power


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
    df = load_raw_timeseries()
    print(df)
    df = normalize_power(df)
    print(df)
    df = discretize_power(df)
    print(df)
    plot_state_distribution(df)


if __name__ == "__main__":
    main()
