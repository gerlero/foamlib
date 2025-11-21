# ruff: noqa: D100
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel
except ImportError as e:
    msg = "The preprocessing module requires extra dependencies. Install 'foamlib[preprocessing]' to use it."
    raise ImportError(msg) from e

from foamlib import FoamFile


class FoamDictInstruction(BaseModel):
    """Class representing an instruction to get a value from a FoamFile."""

    file_name: str | Path
    keys: list[str]

    def get_value(self) -> Any:
        """Get the value from the FoamFile based on the instruction."""
        of_dict = FoamFile(self.file_name)
        return of_dict.get(tuple(self.keys))


class FoamDictAssignment(BaseModel):
    """Class handling the modification of a FoamFile by setting a value for a given instruction."""

    instruction: FoamDictInstruction
    value: Any

    def set_value(self, case_path: Path | None = None) -> FoamFile:
        """Set the value in the FoamFile with the given value and instruction."""
        of_file = Path(self.instruction.file_name)
        if case_path is not None:
            of_file = case_path / of_file
        if not of_file.exists():
            err_msg = f"The file {of_file} does not exist."
            raise FileNotFoundError(err_msg)
        of_dict = FoamFile(of_file)
        of_dict[tuple(self.instruction.keys)] = self.value
        return of_dict
