"""A Python interface for interacting with OpenFOAM."""

from importlib.metadata import version

from ._cases import (
    AsyncFoamCase,
    AsyncSlurmFoamCase,
    CalledProcessError,
    FoamCase,
    FoamCaseBase,
    FoamCaseRunBase,
)
from ._files import FoamFieldFile, FoamFile

__version__ = version("foamlib")

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
