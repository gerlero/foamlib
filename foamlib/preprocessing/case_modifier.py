# ruff: noqa: D100
from __future__ import annotations

from pathlib import Path  # noqa: TC003

try:
    from pydantic import BaseModel
except ImportError as e:
    msg = "The preprocessing module requires extra dependencies. Install 'foamlib[preprocessing]' to use it."
    raise ImportError(msg) from e

from foamlib import FoamCase
from foamlib.preprocessing.of_dict import FoamDictAssignment  # noqa: TC001


class CaseParameter(BaseModel):
    """Class to represent a parameter for a case."""

    category: str
    name: str


class CaseModifier(BaseModel):
    """Class to handle the modification of a case by setting instruction-value pairs."""

    template_case: Path
    output_case: Path
    key_value_pairs: list[FoamDictAssignment]
    case_parameters: list[CaseParameter]

    def create_case(self) -> FoamCase:
        """Create a new case by copying the template case to the output case directory."""
        of_case = FoamCase(path=self.template_case)
        of_case.copy(dst=self.output_case)

        return of_case

    def modify_case(self) -> FoamCase:
        """Modify the case by setting the instruction-value pairs and saving the case."""
        of_case = FoamCase(path=self.output_case)

        for pair in self.key_value_pairs:
            pair.set_value(case_path=self.output_case)

        with (self.output_case / "case.json").open("w") as json_file:
            json_file.write(self.model_dump_json(indent=4))

        return of_case
