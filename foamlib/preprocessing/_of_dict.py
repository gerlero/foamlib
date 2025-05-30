# ruff: noqa: UP006
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import BaseModel

from foamlib import FoamFile


class FoamDictInstruction(BaseModel):
    file_name: Union[str, Path]
    keys: List[str]

    def get_value(self) -> Any:
        of_dict = FoamFile(self.file_name)
        return of_dict.get(tuple(self.keys))


class FoamDictAssignment(BaseModel):
    instruction: FoamDictInstruction
    value: Any

    def set_value(self, case_path: Optional[Path] = None) -> FoamFile:
        of_file = Path(self.instruction.file_name)
        if case_path is not None:
            of_file = case_path / of_file
        if not of_file.exists():
            err_msg = f"The file {of_file} does not exist."
            raise FileNotFoundError(err_msg)
        of_dict = FoamFile(of_file)
        of_dict[tuple(self.instruction.keys)] = self.value
        return of_dict
