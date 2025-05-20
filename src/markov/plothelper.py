from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def decode_bucket(bid: int) -> Tuple[int, int, int]:
    month = bid // 192 + 1
    weekend = (bid % 192) // 96
    quarter = bid % 96
    return month, weekend, quarter


def plot_transition_matrix(P: np.ndarray, bid: int) -> None:
    month, weekend, quarter = decode_bucket(bid)

    plt.figure(figsize=(5, 4))
    plt.imshow(P, origin="lower")
    plt.colorbar(label="P(i → j)")

    plt.title(
        f"Bucket {bid}  (Monat {month}, "
        f"{'WE' if weekend else 'WD'}, Q{quarter:02d})"
    )

    plt.xlabel("Folge‑State j")
    plt.ylabel("Ausgangs‑State i")
    plt.xticks(range(P.shape[0]))
    plt.yticks(range(P.shape[0]))
    plt.tight_layout()
    plt.show()
