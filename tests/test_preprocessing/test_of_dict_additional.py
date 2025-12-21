"""Tests for foamlib.preprocessing.of_dict module - additional coverage."""

from pathlib import Path

import pytest
from foamlib import FoamFile
from foamlib.preprocessing.of_dict import FoamDictAssignment, FoamDictInstruction

DICT_FILE = Path("tests/test_preprocessing/dictionaries/testDictionary")


def test_set_value_with_nonexistent_file() -> None:
    """Test set_value with a nonexistent file raises FileNotFoundError."""
    instruction = FoamDictInstruction(
        file_name=Path("nonexistent/file.txt"), keys=["test1"]
    )
    assignment = FoamDictAssignment(instruction=instruction, value=123)

    with pytest.raises(FileNotFoundError, match=r"nonexistent\/file.txt"):
        assignment.set_value()


def test_set_value_with_case_path() -> None:
    """Test set_value with a case_path argument."""
    # First verify the file structure
    case_path = Path("tests/test_preprocessing")
    instruction = FoamDictInstruction(
        file_name=Path("dictionaries/testDictionary"), keys=["test1"]
    )
    assignment = FoamDictAssignment(instruction=instruction, value=999)

    # Set the value with case_path
    result = assignment.set_value(case_path=case_path)
    assert result.get("test1") == 999

    # Reset the value
    of_dict = FoamFile(case_path / "dictionaries/testDictionary")
    of_dict["test1"] = 1  # Reset to original value
