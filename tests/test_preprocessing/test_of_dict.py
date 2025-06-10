from __future__ import annotations

from pathlib import Path

from foamlib import FoamFile
from foamlib.preprocessing.of_dict import FoamDictAssignment, FoamDictInstruction

DICT_FILE = Path("tests/test_preprocessing/dictionaries/testDictionary")


def test_file_key() -> None:
    """Test the FileKey model."""
    file_key_test1 = FoamDictInstruction(file_name=DICT_FILE, keys=["test1"])
    assert file_key_test1.file_name == Path(DICT_FILE)
    assert file_key_test1.keys == ["test1"]

    assert file_key_test1.get_value() == 1


def test_file_subkey() -> None:
    """Test the FileKey model."""
    file_key_test1 = FoamDictInstruction(
        file_name=DICT_FILE, keys=["subDict", "subSubDict", "test1"]
    )
    assert file_key_test1.file_name == Path(DICT_FILE)
    assert file_key_test1.keys == ["subDict", "subSubDict", "test1"]

    assert file_key_test1.get_value() == 3


def test_file_subkey_value() -> None:
    value_pair = FoamDictAssignment(
        instruction=FoamDictInstruction(
            file_name=DICT_FILE, keys=["subDict", "subSubDict", "test1"]
        ),
        value=123,
    )
    assert value_pair.instruction.file_name == Path(DICT_FILE)
    assert value_pair.instruction.keys == ["subDict", "subSubDict", "test1"]
    assert value_pair.value == 123

    updated_ofDict = value_pair.set_value()

    assert updated_ofDict.get(("subDict", "subSubDict", "test1")) == 123

    reset = FoamFile(DICT_FILE)
    reset[("subDict", "subSubDict", "test1")] = 3

    assert reset.get(("subDict", "subSubDict", "test1")) == 3
