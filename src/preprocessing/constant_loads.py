import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONSTANT_LOADS_FILE = ROOT_DIR / "constant_loads.yml"


def load_constant_loads(
    path: Path | None = None,
) -> dict[str, list[tuple[float, int, int]]]:
    constant_loads_path = CONSTANT_LOADS_FILE if path is None else path
    if not constant_loads_path.exists():
        return {}

    with open(constant_loads_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("constant loads must be a mapping of file stems to lists.")

    result: dict[str, list[tuple[float, int, int]]] = {}
    for key, triples in data.items():
        if not isinstance(key, str):
            raise ValueError(f"constant loads key {key!r} must be a string.")
        if not isinstance(triples, list):
            raise ValueError(f"constant loads entry {key!r} must be a list.")

        normalized_triples: list[tuple[float, int, int]] = []
        for index, triple in enumerate(triples):
            if (
                not isinstance(triple, Sequence)
                or isinstance(triple, str | bytes)
                or len(triple) != 3
            ):
                raise ValueError(
                    f"constant loads entry {key!r} triple {index} must contain "
                    "exactly 3 elements."
                )

            value, second, third = triple
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"constant loads entry {key!r} triple {index} element 0 "
                    "must be a number."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"constant loads entry {key!r} triple {index} element 0 "
                    "must be finite."
                )
            if isinstance(second, bool) or not isinstance(second, int):
                raise ValueError(
                    f"constant loads entry {key!r} triple {index} element 1 "
                    "must be an integer."
                )
            if isinstance(third, bool) or not isinstance(third, int):
                raise ValueError(
                    f"constant loads entry {key!r} triple {index} element 2 "
                    "must be an integer."
                )

            normalized_triples.append((float(value), second, third))

        result[key] = normalized_triples

    return result
