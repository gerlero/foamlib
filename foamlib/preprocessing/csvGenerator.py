from __future__ import annotations
from pathlib import Path
import pandas as pd
from foamlib.preprocessing._caseModifier import CaseModifier, CaseIdentifier
from foamlib.preprocessing._ofDict import FileKey, KeyValuePair
from foamlib.preprocessing.parameterStudy import ParameterStudy



class CSVGenerator:

    def __init__(self, csv_file: str, template_case: Path, output_folder: Path = "Cases") -> None:
        self.csv_file = csv_file
        self.template_case = template_case
        self.output_folder = output_folder
        
    def create_study(self) -> ParameterStudy:
        
        parastudy = pd.read_csv(self.csv_file).to_dict(orient="records")

        cases = []
        for of_case in parastudy:

            case_mod = CaseModifier(
                template_case=self.template_case,
                output_case=self.output_folder / of_case["case_name"],
                key_value_pairs=[
                    KeyValuePair(
                        instruction=FileKey(file_name=Path("system/simulationsParameters"), keys=[key]),
                        value=value
                    ) for key, value in of_case.items()
                ],
                case_identifier=[CaseIdentifier(category="default", name=of_case["case_name"])]
            )
            cases.append(case_mod)
        
        return ParameterStudy(cases=cases)