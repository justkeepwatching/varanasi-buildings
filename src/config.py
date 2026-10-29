"""Load config/config.yaml and define the project's folder layout.

Every other module imports `load_config()` and `PATHS` from here, so file
locations are defined once and never hard-coded in several places.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


class Paths:
    """All folders used by the pipeline, relative to the project root."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.config = root / "config" / "config.yaml"
        self.raw = root / "data" / "raw"
        self.interim = root / "data" / "interim"
        self.processed = root / "data" / "processed"
        self.external = root / "data" / "external"
        self.tiles = self.processed / "tiles"
        self.models = root / "models"
        self.outputs = root / "outputs"
        self.maps = self.outputs / "maps"
        self.metrics = self.outputs / "metrics"
        self.figures = self.outputs / "figures"
        self.vectors = self.outputs / "vectors"
        self.reports = self.outputs / "reports"
        self.rasters = self.outputs / "rasters"

    def make_dirs(self) -> None:
        for p in vars(self).values():
            if isinstance(p, Path) and p.suffix == "":
                p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def load_config(path: Path | str | None = None) -> dict:
    """Read the YAML configuration into a plain dictionary."""
    path = Path(path) if path else PATHS.config
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
