"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.6.3"

from ._cases import (
    AsyncFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
    FoamCaseRunBase,
)
from ._files import FoamFieldFile, FoamFile, FoamFileBase

__all__ = [
    "AsyncFoamCase",
    "CalledProcessError",
    "FoamFile",
    "FoamCase",
    "FoamCaseRunBase",
    "FoamFieldFile",
    "FoamCaseBase",
    "FoamFileBase",
]
