from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def experiment(self) -> Dict[str, Any]:
        return self.raw.get("experiment", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def train(self) -> Dict[str, Any]:
        return self.raw.get("train", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_class_names(root_dir: str | Path) -> List[str]:
    root_dir = Path(root_dir)
    class_names = [p.name for p in root_dir.iterdir() if p.is_dir()]
    class_names.sort()
    return class_names
