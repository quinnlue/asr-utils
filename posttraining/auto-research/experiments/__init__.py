"""
Auto-discovery for experiments.

Drop any .py file in this folder that exposes:
    NAME: str          – human-readable experiment name
    transform(df) -> df – takes a DataFrame, returns a (possibly modified) copy

The runner will pick it up automatically.
"""

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Callable

import pandas as pd


class Experiment:
    """Lightweight wrapper around a discovered experiment module."""

    def __init__(self, name: str, transform: Callable[[pd.DataFrame], pd.DataFrame]):
        self.name = name
        self.transform = transform

    def __repr__(self) -> str:
        return f"Experiment({self.name!r})"


def discover() -> list[Experiment]:
    """Import every sibling module and return an Experiment for each one."""
    package_dir = Path(__file__).resolve().parent
    experiments: list[Experiment] = []

    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_"):
            continue

        module: ModuleType = importlib.import_module(f".{info.name}", package=__package__)

        # Each experiment module must expose a `transform` callable.
        transform_fn = getattr(module, "transform", None)
        if transform_fn is None:
            continue

        name = getattr(module, "NAME", info.name)
        experiments.append(Experiment(name=name, transform=transform_fn))

    # Sort: baseline first, then alphabetically by name
    experiments.sort(key=lambda e: (e.name != "baseline", e.name))
    return experiments
