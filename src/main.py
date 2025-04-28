from src.preprocessing.loader import load_raw_timeseries
from src.preprocessing.scaling import normalize_power


def main():
    df = load_raw_timeseries()
    print(df)
    df = normalize_power(df, col="power")
    print(df)


if __name__ == "__main__":
    main()
