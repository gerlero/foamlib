"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.4.4"

from ._cases import (
    AsyncFoamCase,
    CalledProcessError,
    CalledProcessWarning,
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
    "CalledProcessWarning",
]
