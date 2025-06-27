import os
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


ROOT_DIR = Path(__file__).resolve().parents[1]

cfg_name = os.getenv("APP_CONFIG", "config.yml")

cfg_path = Path(cfg_name)
if not cfg_path.is_absolute():
    cfg_path = ROOT_DIR / cfg_name

cfg_path = cfg_path.resolve()
if not cfg_path.exists():
    raise FileNotFoundError(f"Config file not found: {cfg_path}")

CONFIG: dict = load_config(cfg_path)
