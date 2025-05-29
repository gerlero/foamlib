from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from typing import List

from foamlib import FoamCase
from foamlib.preprocessing._of_dict import KeyValuePair


class CaseParameter(BaseModel):
    category: str
    name: str


class CaseModifier(BaseModel):
    template_case: Path
    output_case: Path
    key_value_pairs: List[KeyValuePair]
    case_parameters: List[CaseParameter]

    def create_case(self) -> FoamCase:
        of_case = FoamCase(path=self.template_case)
        of_case.copy(dst=self.output_case)

        return of_case

    def modify_case(self) -> FoamCase:
        of_case = FoamCase(path=self.output_case)

        for pair in self.key_value_pairs:
            pair.set_value(case_path=self.output_case)

        with open(self.output_case / "case.json", "w") as json_file:
            json_file.write(self.model_dump_json(indent=4))

        return of_case
