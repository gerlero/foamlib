# ruff: noqa: N802, D100
from __future__ import annotations

from pathlib import Path

from foamlib.preprocessing.of_dict import FoamDictInstruction


def simulationParameters(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for simulationParameters."""
    return FoamDictInstruction(
        file_name=Path("system/simulationParameters"),
        keys=keys,
    )


def controlDict(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for controlDict."""
    return FoamDictInstruction(
        file_name=Path("system/controlDict"),
        keys=keys,
    )


def fvSchemes(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for fvSchemes."""
    return FoamDictInstruction(
        file_name=Path("system/fvSchemes"),
        keys=keys,
    )


def fvSolution(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for fvSolution."""
    return FoamDictInstruction(
        file_name=Path("system/fvSolution"),
        keys=keys,
    )
