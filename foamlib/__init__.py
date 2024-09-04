"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.3.23"

from ._cases import (
    AsyncFoamCase,
    CalledProcessError,
    CalledProcessWarning,
    FoamCase,
    FoamCaseBase,
)
from ._files import FoamDict, FoamFieldFile, FoamFile

__all__ = [
    "FoamCase",
    "AsyncFoamCase",
    "FoamCaseBase",
    "FoamFile",
    "FoamFieldFile",
    "FoamDict",
    "CalledProcessError",
    "CalledProcessWarning",
]
