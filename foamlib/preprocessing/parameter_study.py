"""Parameter study module for generating multiple cases based on parameter combinations."""

# ruff: noqa: UP006
from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable, List

import pandas as pd
from pydantic import BaseModel

from foamlib import FoamFile
from foamlib.preprocessing.case_modifier import CaseModifier, CaseParameter
from foamlib.preprocessing.of_dict import FoamDictAssignment, FoamDictInstruction

if TYPE_CHECKING:
    from foamlib.preprocessing.grid_parameter_sweep import GridParameter


class ParameterStudy(BaseModel):
    """Class to handle a parameter study by creating multiple cases based on parameter combinations."""

    cases: List[CaseModifier]

    def create_study(self, study_base_folder: Path = Path()) -> None:
        """Create multiple cases based on the parameter combinations."""
        with (study_base_folder / "parameter_study.json").open("w") as json_file:
            json_file.write(self.model_dump_json(indent=2))

        for of_case in self.cases:
            of_case.create_case()
            of_case.modify_case()

    def __add__(self, other: ParameterStudy) -> ParameterStudy:
        """Combine two ParameterStudy instances."""
        return ParameterStudy(cases=self.cases + other.cases)

def record_generator(
    records: List[dict[Hashable, Any]],
    template_case: str | Path,
    output_folder: str | Path = Path("Cases"),
) -> ParameterStudy:
    """Generate a parameter study based on records.

    Example records: [
        {'case_name': '3DCube_N10', 'Res': 10, 'MeshType': '3DCube', 'Resolution': 'N10'},
        {'case_name': '3DCube_N20', 'Res': 20, 'MeshType': '3DCube', 'Resolution': 'N20'},
        {'case_name': '3DCube_N50', 'Res': 50, 'MeshType': '3DCube', 'Resolution': 'N50'},
    ]
    """
    parameter = FoamFile(
        Path(template_case) / "system" / "simulationParameters"
    ).as_dict()
    parameter_keys = set(parameter.keys())
    case_keys = set(records[0].keys())
    category_keys = case_keys - parameter_keys - {"case_name"}

    cases = []
    for of_case in records:
        case_mod = CaseModifier(
            template_case=Path(template_case),
            output_case=Path(output_folder) / of_case["case_name"],
            key_value_pairs=[
                FoamDictAssignment(
                    instruction=FoamDictInstruction(
                        file_name=Path("system/simulationParameters"), keys=[str(key)]
                    ),
                    value=value,
                )
                for key, value in of_case.items()
                if key in parameter_keys
            ],
            case_parameters=[
                CaseParameter(category=str(key), name=str(of_case[str(key)]))
                for key, value in of_case.items()
                if key in category_keys
            ],
        )
        cases.append(case_mod)

    return ParameterStudy(cases=cases)

def csv_generator(
    csv_file: str | Path,
    template_case: str | Path,
    output_folder: str | Path = Path("Cases"),
) -> ParameterStudy:
    """Generate a parameter study from a CSV file."""
    parastudy = pd.read_csv(str(Path(csv_file))).to_dict(orient="records")
    return record_generator(
        parastudy,
        template_case,
        output_folder,
    )


def grid_generator(
    parameters: List[GridParameter],
    template_case: str | Path,
    output_folder: str | Path = Path("Cases"),
) -> ParameterStudy:
    """Generate a parameter study based on grid parameters."""
    cases = []

    categories = [param.parameter_name for param in parameters]
    case_instructions = [ins for param in parameters for ins in param.modify_dict]
    case_parameters = itertools.product(*[param.parameters for param in parameters])

    for case_parameter in case_parameters:
        flattened_parameters = list(
            itertools.chain.from_iterable(val.values for val in case_parameter)  # noqa: PD011
        )
        case_name = [val.name for val in case_parameter]
        case_modifications = [
            FoamDictAssignment(
                instruction=case_instructions[i],
                value=flattened_parameters[i],
            )
            for i in range(len(case_instructions))
        ]
        case_mod = CaseModifier(
            template_case=Path(template_case),
            output_case=Path(output_folder) / "_".join(case_name),
            key_value_pairs=case_modifications,
            case_parameters=[
                CaseParameter(category=categories[i], name=str(case_parameter[i].name))
                for i in range(len(case_parameter))
            ],
        )
        cases.append(case_mod)

    return ParameterStudy(cases=cases)
