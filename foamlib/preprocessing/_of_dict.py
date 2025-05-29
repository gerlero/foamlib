from __future__ import annotations

from typing import List
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from foamlib import FoamFile


class FileKey(BaseModel):
    file_name: Path
    keys: List[str]

    def get_value(self) -> Any:
        of_dict = FoamFile(self.file_name)
        value = of_dict
        for key in self.keys:
            value = value.get(key)
            if value is None:
                break
        return value


class KeyValuePair(BaseModel):
    instruction: FileKey
    value: Any

    def set_value(self, case_path: Optional[Path] = None) -> FoamFile:
        of_file = self.instruction.file_name
        if case_path is not None:
            of_file = case_path / of_file
        if not of_file.exists():
            err_msg = f"The file {of_file} does not exist."
            raise FileNotFoundError(err_msg)
        of_dict = FoamFile(of_file)
        current_dict = of_dict
        for key in self.instruction.keys[:-1]:
            current_dict = current_dict.get(key, {})
        current_dict[self.instruction.keys[-1]] = self.value
        return of_dict
