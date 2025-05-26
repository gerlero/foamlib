from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from foamlib.preprocessing.parameter_study import csv_generator

CSV_FILE = "tests/test_preprocessing/test_parastudy.csv"
OUTPUT_FOLDER = "tests/test_preprocessing/Cases/"


@pytest.fixture
def output_folder() -> Path:
    """Fixture to clean up the output case folder after the test."""
    output_cases = Path(OUTPUT_FOLDER)
    yield output_cases  # Provide the folder path to the test
    if output_cases.exists():
        shutil.rmtree(output_cases)  # Remove the folder after the test


def test_csv_generator(output_folder: Path) -> None:
    """Test the CSVGenerator model."""
    template_case = Path("tests/test_preprocessing/templates/damBreak")

    study = csv_generator(
        csv_file=CSV_FILE, template_case=template_case, output_folder=output_folder
    )

    assert len(study.cases) == 2  # Assuming the CSV has 3 cases

    study.create_study()

    for case in study.cases:
        assert case.output_case.exists()
        assert len(case.key_value_pairs) > 0
        assert case.case_identifier[0].category == "grid"
        assert case.case_identifier[1].category == "initHeight"
        assert case.case_identifier[0].name in ["res1"]
        assert case.case_identifier[1].name in ["height_02", "height_03"]
