"""Parameter study module for generating multiple cases based on parameter combinations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from typing import List

from foamlib import FoamFile
from foamlib.preprocessing._case_modifier import CaseModifier, CaseParameter
from foamlib.preprocessing._of_dict import FileKey, KeyValuePair


class ParameterStudy(BaseModel):
    """Class to handle a parameter study by creating multiple cases based on parameter combinations."""

    cases: List[CaseModifier]

    def create_study(self) -> None:
        """Create multiple cases based on the parameter combinations."""
        for of_case in self.cases:
            of_case.create_case()
            of_case.modify_case()


def csv_generator(
    csv_file: str, template_case: Path, output_folder: Path = "Cases"
) -> ParameterStudy:
    """Generate a parameter study from a CSV file."""
    parastudy = pd.read_csv(csv_file).to_dict(orient="records")
    parameter = FoamFile(template_case / "system" / "simulationsParameters").as_dict()
    parameter_keys = set(parameter.keys())
    case_keys = set(parastudy[0].keys())
    category_keys = case_keys - parameter_keys - {"case_name"}

    cases = []
    for of_case in parastudy:
        case_mod = CaseModifier(
            template_case=template_case,
            output_case=output_folder / of_case["case_name"],
            key_value_pairs=[
                KeyValuePair(
                    instruction=FileKey(
                        file_name=Path("system/simulationsParameters"), keys=[key]
                    ),
                    value=value,
                )
                for key, value in of_case.items()
                if key in parameter_keys
            ],
            case_parameters=[
                CaseParameter(category=key, name=of_case[key])
                for key, value in of_case.items()
                if key in category_keys
            ],
        )
        cases.append(case_mod)

    return ParameterStudy(cases=cases)
