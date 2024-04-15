"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.3.0"

from ._cases import AsyncFoamCase, FoamCase, FoamCaseBase
from ._files import FoamDict, FoamFieldFile, FoamFile
from ._util import CalledProcessError

__all__ = [
    "FoamCase",
    "AsyncFoamCase",
    "FoamCaseBase",
    "FoamFile",
    "FoamFieldFile",
    "FoamDict",
    "CalledProcessError",
]
