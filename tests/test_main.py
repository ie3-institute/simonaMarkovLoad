from pathlib import Path

from src.main import _discover_pools, _pool_output_path


def test_pool_output_path_inserts_pool_name_before_suffix():
    base = Path("out/psdm_model.json")

    assert _pool_output_path(base, "north") == Path("out/psdm_model_north.json")


def test_discover_pools_returns_sorted_subdirs_and_loose_csv_count(tmp_path):
    north = tmp_path / "north"
    south = tmp_path / "south"
    south.mkdir()
    north.mkdir()
    (north / "nested").mkdir()
    (tmp_path / "loose.csv").touch()
    (tmp_path / "notes.txt").touch()

    pools, loose_csv_count = _discover_pools(tmp_path)

    assert pools == [north, south]
    assert loose_csv_count == 1
