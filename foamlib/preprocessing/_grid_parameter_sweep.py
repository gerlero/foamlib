# ruff: noqa: UP006
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel

from foamlib.preprocessing._of_dict import FoamDictInstruction


class CaseParameter(BaseModel):
    name: str
    values: List[Any]


class GridParameter(BaseModel):
    parameter_name: str
    modify_dict: List[FoamDictInstruction]
    parameters: List[CaseParameter]

    def case_names(self) -> List[str]:
        """Return the names of the cases."""
        return [param.name for param in self.parameters]
