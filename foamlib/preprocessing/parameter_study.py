"""Parameter study module for generating multiple cases based on parameter combinations."""

# ruff: noqa: UP006
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
from pydantic import BaseModel

from foamlib import FoamFile
from foamlib.preprocessing._case_modifier import CaseModifier, CaseParameter
from foamlib.preprocessing._of_dict import FoamDictAssignment, FoamDictInstruction


class ParameterStudy(BaseModel):
    """Class to handle a parameter study by creating multiple cases based on parameter combinations."""

    cases: List[CaseModifier]

    def create_study(self, study_base_folder: Path = Path()) -> None:
        """Create multiple cases based on the parameter combinations."""
        with open(study_base_folder / "parameter_study.json", "w") as json_file:
            json_file.write(self.model_dump_json(indent=2))

        for of_case in self.cases:
            of_case.create_case()
            of_case.modify_case()


def csv_generator(
    csv_file: str,
    template_case: Union[str, Path],
    output_folder: Union[str, Path] = Path("Cases"),
) -> ParameterStudy:
    """Generate a parameter study from a CSV file."""
    parastudy = pd.read_csv(csv_file).to_dict(orient="records")
    parameter = FoamFile(
        Path(template_case) / "system" / "simulationsParameters"
    ).as_dict()
    parameter_keys = set(parameter.keys())
    case_keys = set(parastudy[0].keys())
    category_keys = case_keys - parameter_keys - {"case_name"}

    cases = []
    for of_case in parastudy:
        case_mod = CaseModifier(
            template_case=Path(template_case),
            output_case=Path(output_folder) / of_case["case_name"],
            key_value_pairs=[
                FoamDictAssignment(
                    instruction=FoamDictInstruction(
                        file_name=Path("system/simulationsParameters"), keys=[str(key)]
                    ),
                    value=value,
                )
                for key, value in of_case.items()
                if key in parameter_keys
            ],
            case_parameters=[
                CaseParameter(category=str(key), name=of_case[str(key)])
                for key, value in of_case.items()
                if key in category_keys
            ],
        )
        cases.append(case_mod)

    return ParameterStudy(cases=cases)
