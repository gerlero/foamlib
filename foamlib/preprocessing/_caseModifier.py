from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

from foamlib import FoamCase
from foamlib.preprocessing._ofDict import FileKey, KeyValuePair

class CaseIdentifier(BaseModel):
    category: str
    name: str

    def __str__(self) -> str:
        return f"CaseIdentifier(template_case={self.template_case}, output_case={self.output_case})"

    def __repr__(self) -> str:
        return self.__str__()

class CaseModifier(BaseModel):
    template_case: Path
    output_case: Path
    key_value_pairs: list[KeyValuePair]
    case_identifier: list[CaseIdentifier]


    def create_case(self) -> FoamCase:
        of_case = FoamCase(path=self.template_case)
        of_case.clone(dst=self.output_case)

        return of_case
    
    def modify_case(self) -> FoamCase:
        of_case = FoamCase(path=self.output_case)

        for pair in self.key_value_pairs:
            pair.set_value(case_path=self.output_case)

        with open(self.output_case / "case.json", "w") as json_file:
            json_file.write(self.model_dump_json(indent=4))

        return of_case