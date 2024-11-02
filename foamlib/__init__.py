"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.6.12"

from ._cases import (
    AsyncFoamCase,
    AsyncSlurmFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
    FoamCaseRunBase,
)
from ._files import FoamFieldFile, FoamFile, FoamFileBase

__all__ = [
    "AsyncFoamCase",
    "AsyncSlurmFoamCase",
    "CalledProcessError",
    "FoamFile",
    "FoamCase",
    "FoamCaseBase",
    "FoamCaseRunBase",
    "FoamFieldFile",
    "FoamFileBase",
]
