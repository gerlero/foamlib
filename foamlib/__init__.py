"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.6.0"

from ._cases import (
    AsyncFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
)
from ._files import FoamFieldFile, FoamFile, FoamFileBase

__all__ = [
    "FoamCase",
    "AsyncFoamCase",
    "FoamCaseBase",
    "FoamFile",
    "FoamFieldFile",
    "FoamFileBase",
    "CalledProcessError",
]
