# ruff: noqa: UP006, D100
from __future__ import annotations

from typing import Any, List

try:
    from pydantic import BaseModel
except ImportError as e:
    msg = "The preprocessing module requires extra dependencies. Install 'foamlib[preprocessing]' to use it."
    raise ImportError(msg) from e

from foamlib.preprocessing.of_dict import FoamDictInstruction  # noqa: TC001


class CaseParameter(BaseModel):
    """Class to represent a parameter for a case."""

    name: str
    values: List[Any]


class GridParameter(BaseModel):
    """Class to handle a grid parameter sweep by creating multiple cases based on parameter combinations."""

    parameter_name: str
    modify_dict: List[FoamDictInstruction]
    parameters: List[CaseParameter]

    def case_names(self) -> List[str]:
        """Return the names of the cases."""
        return [param.name for param in self.parameters]
