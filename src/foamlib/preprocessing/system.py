# ruff: noqa: D100
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

from foamlib.preprocessing.of_dict import FoamDictInstruction


def simulation_parameters(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for simulationParameters."""
    return FoamDictInstruction(
        file_name=Path("system/simulationParameters"),
        keys=keys,
    )


@deprecated("Use 'simulation_parameters' instead")
def simulationParameters(keys: list[str]) -> FoamDictInstruction:  # noqa: N802
    """
    Alias for :func:`simulation_parameters`.

    Deprecated since version 1.3.0: use :func:`simulation_parameters` instead.
    """
    return simulation_parameters(keys)


def control_dict(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for controlDict."""
    return FoamDictInstruction(
        file_name=Path("system/controlDict"),
        keys=keys,
    )


@deprecated("Use 'control_dict' instead")
def controlDict(keys: list[str]) -> FoamDictInstruction:  # noqa: N802
    """
    Alias for :func:`control_dict`.

    Deprecated since version 1.3.0: use :func:`control_dict` instead.
    """
    return control_dict(keys)


def fv_schemes(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for fvSchemes."""
    return FoamDictInstruction(
        file_name=Path("system/fvSchemes"),
        keys=keys,
    )


@deprecated("Use 'fv_schemes' instead")
def fvSchemes(keys: list[str]) -> FoamDictInstruction:  # noqa: N802
    """
    Alias for :func:`fv_schemes`.

    Deprecated since version 1.3.0: use :func:`fv_schemes` instead.
    """
    return fv_schemes(keys)


def fv_solution(keys: list[str]) -> FoamDictInstruction:
    """Return the FoamDictInstruction for fvSolution."""
    return FoamDictInstruction(
        file_name=Path("system/fvSolution"),
        keys=keys,
    )


@deprecated("Use 'fv_solution' instead")
def fvSolution(keys: list[str]) -> FoamDictInstruction:  # noqa: N802
    """
    Alias for :func:`fv_solution`.

    Deprecated since version 1.3.0: use :func:`fv_solution` instead.
    """
    return fv_solution(keys)
