from pathlib import Path
from foamlib import FoamFile
from foamlib.preprocessing._ofDict import FileKey, KeyValuePair


DICT_FILE = "tests/test_preprocessing/dictionaries/testDictionary"

def test_file_key() -> None:
    """Test the FileKey model."""
    file_key = FileKey(file_name=DICT_FILE, keys=["test1"])
    assert file_key.file_name == Path(DICT_FILE)
    assert file_key.keys == ["test1"]

    ofDict = FoamFile(file_key.file_name)

    assert file_key.get_value() == 1