from __future__ import annotations

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator

import pytest
from foamlib import FoamCase
from foamlib.preprocessing.case_modifier import CaseModifier, CaseParameter
from foamlib.preprocessing.of_dict import FoamDictAssignment, FoamDictInstruction

OUTPUT_CASE = "tests/test_preprocessing/modifiedCase"


@pytest.fixture
def output_case() -> Generator[Path, None, None]:
    """Fixture to clean up the output case folder after the test."""
    output_case = Path(OUTPUT_CASE)
    yield output_case  # Provide the folder path to the test
    if output_case.exists():
        shutil.rmtree(output_case)  # Remove the folder after the test


def test_case_modifier(output_case: Path) -> None:
    """Test the CaseModifier model."""
    template_case = Path("tests/test_preprocessing/templates/damBreak")
    key_value_pairs = [
        FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=Path("system/controlDict"), keys=["endTime"]
            ),
            value=42,
        ),
        FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=Path("constant/transportProperties"),
                keys=["water", "transportModel"],
            ),
            value="asdf",
        ),
    ]

    case_modifier = CaseModifier(
        template_case=template_case,
        key_value_pairs=key_value_pairs,
        output_case=output_case,
        case_parameters=[
            CaseParameter(
                category="testCategory",
                name="testName",
            )
        ],
    )

    case_modifier.create_case()

    assert Path(case_modifier.output_case).exists()

    # Check if the key_value_pairs are correctly set
    assert len(case_modifier.key_value_pairs) == 2
    assert case_modifier.key_value_pairs[0].instruction.file_name == Path(
        "system/controlDict"
    )
    assert case_modifier.key_value_pairs[0].instruction.keys == ["endTime"]
    assert case_modifier.key_value_pairs[0].value == 42
    assert case_modifier.key_value_pairs[1].instruction.file_name == Path(
        "constant/transportProperties"
    )
    assert case_modifier.key_value_pairs[1].instruction.keys == [
        "water",
        "transportModel",
    ]
    assert case_modifier.key_value_pairs[1].value == "asdf"
    assert case_modifier.case_parameters[0].category == "testCategory"
    assert case_modifier.case_parameters[0].name == "testName"

    case_modifier.modify_case()
    assert Path(case_modifier.output_case / "case.json").exists()
    # Check if the modified case has the expected values
    control_dict = case_modifier.output_case / "system/controlDict"
    transport_properties = case_modifier.output_case / "constant/transportProperties"
    assert control_dict.exists()
    assert transport_properties.exists()

    of_case = FoamCase(path=case_modifier.output_case)
    assert of_case.control_dict.get("endTime") == 42
    assert of_case.transport_properties.get(("water", "transportModel")) == "asdf"
