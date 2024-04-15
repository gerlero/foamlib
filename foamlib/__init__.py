"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.2.10"

from ._cases import AsyncFoamCase, FoamCase, FoamCaseBase
from ._files import FoamDict, FoamFieldFile, FoamFile

__all__ = [
    "FoamCase",
    "AsyncFoamCase",
    "FoamCaseBase",
    "FoamFile",
    "FoamFieldFile",
    "FoamDict",
]
