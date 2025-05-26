from __future__ import annotations
from typing import Any
from foamlib import FoamFile
from pathlib import Path
from pydantic import BaseModel, Field


class FileKey(BaseModel):
    file_name: Path 
    keys: list[str]

    def get_value(self) -> Any:
        ofDict = FoamFile(self.file_name)
        return ofDict.get(self.keys[0])


class KeyValuePair(BaseModel):
    fileWithKey: FileKey
    value: Any