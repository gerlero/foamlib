# ruff: noqa: D100
from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel
except ImportError as e:
    msg = "The preprocessing module requires extra dependencies. Install 'foamlib[preprocessing]' to use it."
    raise ImportError(msg) from e

from foamlib.preprocessing.of_dict import FoamDictInstruction  # noqa: TC001


class CaseParameter(BaseModel):
    """Class to represent a parameter for a case."""

    name: str
    values: list[Any]


class GridParameter(BaseModel):
    """Class to handle a grid parameter sweep by creating multiple cases based on parameter combinations."""

    parameter_name: str
    modify_dict: list[FoamDictInstruction]
    parameters: list[CaseParameter]

    def case_names(self) -> list[str]:
        """Return the names of the cases."""
        return [param.name for param in self.parameters]
