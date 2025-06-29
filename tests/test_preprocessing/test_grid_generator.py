# ruff: noqa: ERA001
from __future__ import annotations

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator

import pytest
from foamlib.preprocessing.grid_parameter_sweep import CaseParameter, GridParameter
from foamlib.preprocessing.of_dict import FoamDictInstruction
from foamlib.preprocessing.parameter_study import grid_generator
from foamlib.preprocessing.system import simulationParameters

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


def grid_parameters(scale: float) -> list[int]:
    return [
        int(23 * scale),
        int(8 * scale),
        int(19 * scale),
        int(42 * scale),
        int(4 * scale),
    ]


def test_grid_parameter() -> None:
    grid = GridParameter(
        parameter_name="grid",
        # generate 5 instructions in system/simulationParameters with the key1..5
        # This is simulationParameters is identical to the following:
        #      FoamDictInstruction(
        #         file_name=Path("system/simulationParameters"),
        #         keys=[f"res{i}"],
        #      )
        modify_dict=[simulationParameters(keys=[f"res{i}"]) for i in range(1, 6)],
        parameters=[
            CaseParameter(name="coarse", values=grid_parameters(1)),
            CaseParameter(name="mid", values=grid_parameters(2)),
            CaseParameter(name="fine", values=grid_parameters(4)),
        ],
    )

    assert grid.parameter_name == "grid"
    assert grid.case_names() == ["coarse", "mid", "fine"]
    assert len(grid.modify_dict) == 5
    assert grid.modify_dict[0].file_name == Path("system/simulationParameters")
    assert grid.modify_dict[0].keys == ["res1"]
    assert grid.parameters[0].values == grid_parameters(1)
    assert grid.parameters[0].name == "coarse"
    assert len(grid.parameters) == 3

    init_height = GridParameter(
        parameter_name="initHeight",
        modify_dict=[
            FoamDictInstruction(
                file_name=Path("system/simulationParameters"),
                keys=["initHeight"],
            )
        ],
        parameters=[
            CaseParameter(name="height_02", values=[0.2]),
            CaseParameter(name="height_03", values=[0.3]),
        ],
    )

    assert init_height.parameter_name == "initHeight"
    assert init_height.case_names() == ["height_02", "height_03"]
    assert len(init_height.modify_dict) == 1
    assert init_height.modify_dict[0].file_name == Path("system/simulationParameters")
    assert init_height.modify_dict[0].keys == ["initHeight"]
    assert init_height.parameters[0].values == [0.2]
    assert init_height.parameters[0].name == "height_02"


def test_grid_generator(output_folder: Path) -> None:
    """Test the CSVGenerator model."""
    template_case = Path("tests/test_preprocessing/templates/damBreak")

    grid = GridParameter(
        parameter_name="grid",
        # generate 5 instructions in system/simulationParameters with the key1..5
        modify_dict=[
            FoamDictInstruction(
                file_name=Path("system/simulationParameters"),
                keys=[f"res{i}"],
            )
            for i in range(1, 6)
        ],
        parameters=[
            CaseParameter(name="coarse", values=grid_parameters(1)),
            CaseParameter(name="mid", values=grid_parameters(2)),
            CaseParameter(name="fine", values=grid_parameters(4)),
        ],
    )

    init_height = GridParameter(
        parameter_name="initHeight",
        modify_dict=[
            FoamDictInstruction(
                file_name=Path("system/simulationParameters"),
                keys=["initHeight"],
            )
        ],
        parameters=[
            CaseParameter(name="height_02", values=[0.2]),
            CaseParameter(name="height_03", values=[0.3]),
            CaseParameter(name="height_04", values=[0.4]),
        ],
    )

    study = grid_generator(
        parameters=[grid, init_height],
        template_case=template_case,
        output_folder=output_folder,
    )

    assert len(study.cases) == 9

    study.create_study(study_base_folder=output_folder.parent)

    assert Path(output_folder.parent / "parameter_study.json").exists()

    for case in study.cases:
        assert case.output_case.exists()
        assert len(case.key_value_pairs) == 6
        assert case.case_parameters[0].category == "grid"
        assert case.case_parameters[1].category == "initHeight"
        assert case.case_parameters[0].name in ["coarse", "mid", "fine"]
        assert case.case_parameters[1].name in ["height_02", "height_03", "height_04"]
