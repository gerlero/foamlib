"""A Python interface for interacting with OpenFOAM."""

__version__ = "1.0.0"

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
