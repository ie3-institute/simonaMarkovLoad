from pathlib import Path

import pytest

from src.preprocessing.constant_loads import load_constant_loads


def test_load_constant_loads_parses_and_normalizes_valid_file(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text(
        "SM_00001:\n"
        "  - [330, 20, 6]\n"
        "SM_00002:\n"
        "  - [150.5, 24, 8]\n"
        "  - [80.0, 12, 2]\n",
        encoding="utf-8",
    )

    assert load_constant_loads(path) == {
        "SM_00001": [(330.0, 20, 6)],
        "SM_00002": [(150.5, 24, 8), (80.0, 12, 2)],
    }


def test_load_constant_loads_missing_file_returns_empty_mapping(tmp_path: Path):
    assert load_constant_loads(tmp_path / "missing.yml") == {}


def test_load_constant_loads_empty_file_returns_empty_mapping(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("", encoding="utf-8")

    assert load_constant_loads(path) == {}


def test_load_constant_loads_null_entry_returns_empty_list(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("SM_00001:\n", encoding="utf-8")

    assert load_constant_loads(path) == {"SM_00001": []}


def test_file_stem_without_entry_uses_mapping_default(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("SM_00001:\n  - [330, 20, 6]\n", encoding="utf-8")

    constant_loads = load_constant_loads(path)

    assert constant_loads.get("SM_99999", []) == []


def test_load_constant_loads_rejects_non_mapping_top_level(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("- [330, 20, 6]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_constant_loads(path)


@pytest.mark.parametrize(
    "entry",
    [
        "[330, 20]",
        "[not-a-number, 20, 6]",
        "[true, 20, 6]",
        "[330, 20.5, 6]",
        "[330, 20, 6.5]",
        "[330, false, 6]",
        "[330, 20, true]",
        "[.inf, 20, 6]",
    ],
)
def test_load_constant_loads_rejects_invalid_triples(tmp_path: Path, entry: str):
    path = tmp_path / "constant_loads.yml"
    path.write_text(f"SM_BAD:\n  - {entry}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SM_BAD"):
        load_constant_loads(path)


def test_load_constant_loads_rejects_non_list_entry(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("SM_BAD: invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SM_BAD"):
        load_constant_loads(path)


def test_load_constant_loads_rejects_non_string_key(tmp_path: Path):
    path = tmp_path / "constant_loads.yml"
    path.write_text("1:\n  - [330, 20, 6]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1"):
        load_constant_loads(path)
