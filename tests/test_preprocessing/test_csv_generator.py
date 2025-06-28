from __future__ import annotations

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator

import pytest
from foamlib.postprocessing.load_tables import functionobject, load_tables
from foamlib.preprocessing.parameter_study import csv_generator

CSV_FILE = "tests/test_preprocessing/test_parastudy.csv"
OUTPUT_FOLDER = "tests/test_preprocessing/Cases/"


@pytest.fixture
def output_folder() -> Generator[Path, None, None]:
    """Fixture to clean up the output case folder after the test."""
    output_cases = Path(OUTPUT_FOLDER)
    yield output_cases  # Provide the folder path to the test
    if output_cases.exists():
        shutil.rmtree(output_cases)  # Remove the folder after the test
        Path(output_cases.parent / "parameter_study.json").unlink()


def test_csv_generator(output_folder: Path) -> None:
    """Test the CSVGenerator model."""
    template_case = Path("tests/test_preprocessing/templates/damBreak")

    study = csv_generator(
        csv_file=CSV_FILE, template_case=template_case, output_folder=output_folder
    )

    assert len(study.cases) == 2  # Assuming the CSV has 3 cases

    study.create_study(study_base_folder=output_folder.parent)

    assert Path(output_folder.parent / "parameter_study.json").exists()

    for case in study.cases:
        assert case.output_case.exists()
        assert len(case.key_value_pairs) > 0
        assert case.case_parameters[0].category == "grid"
        assert case.case_parameters[1].category == "initHeight"
        assert case.case_parameters[0].name in ["res1"]
        assert case.case_parameters[1].name in ["height_02", "height_03"]


def test_post_processing(output_folder: Path) -> None:
    """Test the CSVGenerator model."""
    template_case = Path("tests/test_preprocessing/templates/damBreak")

    study = csv_generator(
        csv_file=CSV_FILE, template_case=template_case, output_folder=output_folder
    )

    assert len(study.cases) == 2  # Assuming the CSV has 2 cases

    study.create_study(study_base_folder=output_folder.parent)

    for case in study.cases:
        assert case.output_case.exists()
        assert len(case.key_value_pairs) > 0

    forces = load_tables(
        source=functionobject(file_name="force.dat", folder="forces"),
        dir_name=output_folder,
    )
    assert forces is not None
    assert forces.columns.tolist() == [
        "Time",
        "total_x",
        "total_y",
        "total_z",
        "pressure_x",
        "pressure_y",
        "pressure_z",
        "viscous_x",
        "viscous_y",
        "viscous_z",
        "grid",
        "initHeight",
    ]

    assert forces is not None
    assert forces["grid"].unique().tolist() == ["res1"]
    assert sorted(forces["initHeight"].unique().tolist()) == ["height_02", "height_03"]
