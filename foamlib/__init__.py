"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.3.12"

from ._cases import AsyncFoamCase, FoamCase, FoamCaseBase
from ._files import FoamDict, FoamFieldFile, FoamFile
from ._util import CalledProcessError, CalledProcessWarning

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
