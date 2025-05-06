"""A Python interface for interacting with OpenFOAM."""

import sys

if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

__version__ = importlib_metadata.version("foamlib")

from ._cases import (
    AsyncFoamCase,
    AsyncSlurmFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
    FoamCaseRunBase,
)
from ._files import FoamFieldFile, FoamFile

__all__ = [
    "AsyncFoamCase",
    "AsyncSlurmFoamCase",
    "CalledProcessError",
    "FoamCase",
    "FoamCaseBase",
    "FoamCaseRunBase",
    "FoamFieldFile",
    "FoamFile",
]
