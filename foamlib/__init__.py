"""A Python interface for interacting with OpenFOAM."""

__version__ = "0.2.9"

from ._cases import AsyncFoamCase, FoamCase, FoamCaseBase
from ._dictionaries import FoamDictionaryBase, FoamFieldFile, FoamFile

__all__ = [
    "FoamCase",
    "AsyncFoamCase",
    "FoamCaseBase",
    "FoamFile",
    "FoamFieldFile",
    "FoamDictionaryBase",
]
