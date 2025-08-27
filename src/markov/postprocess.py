import numpy as np
import pandas as pd

def apply_dwell_time(states: np.ndarray, p_extend: float = 0.7) -> np.ndarray:
    if states.size < 2:
        return states

    out = states.copy()
    hold_next = False
    for t in range(1, len(states)):
        if hold_next:
            out[t] = out[t-1]
            hold_next = False
        elif states[t] != states[t-1] and np.random.rand() < p_extend:
            out[t] = out[t-1]
            hold_next = True
    return out


def two_point_smooth(series: pd.Series) -> pd.Series:
    return 0.5 * (series.shift(1, fill_value=series.iloc[0]) + series)
