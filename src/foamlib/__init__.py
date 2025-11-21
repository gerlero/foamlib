"""A Python interface for interacting with OpenFOAM."""

from importlib.metadata import version

from ._cases import (
    AsyncFoamCase,
    AsyncSlurmFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
)
from ._files import Dimensioned, DimensionSet, FoamFieldFile, FoamFile

__version__ = version("foamlib")

__all__ = [
    "AsyncFoamCase",
    "AsyncSlurmFoamCase",
    "CalledProcessError",
    "DimensionSet",
    "Dimensioned",
    "FoamCase",
    "FoamCaseBase",
    "FoamFieldFile",
    "FoamFile",
]
