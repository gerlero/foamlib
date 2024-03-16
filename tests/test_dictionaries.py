import pytest

from pathlib import Path

from foamlib import FoamFile, FoamDictionary


def test_parse() -> None:
    assert FoamDictionary._parse("1") == 1
    assert FoamDictionary._parse("1.0") == 1.0
    assert FoamDictionary._parse("1.0e-3") == 1.0e-3
    assert FoamDictionary._parse("yes") == True
    assert FoamDictionary._parse("no") == False
    assert FoamDictionary._parse("uniform 1") == 1
    assert FoamDictionary._parse("uniform 1.0") == 1.0
    assert FoamDictionary._parse("uniform 1.0e-3") == 1.0e-3
    assert FoamDictionary._parse("(1.0 2.0 3.0)") == [1.0, 2.0, 3.0]
    assert FoamDictionary._parse("nonuniform List<scalar> 2(1 2)") == [1, 2]
    assert FoamDictionary._parse("3(1 2 3)") == [1, 2, 3]
    assert FoamDictionary._parse("2((1 2 3) (4 5 6))") == [[1, 2, 3], [4, 5, 6]]
    assert FoamDictionary._parse("nonuniform List<vector> 2((1 2 3) (4 5 6))") == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_write_read(tmp_path: Path) -> None:
    path = tmp_path / "testDict"
    path.touch()

    d = FoamFile(path)
    assert d.path == path
    assert not d
    assert len(d) == 0
    assert list(d) == []
    with pytest.raises(KeyError):
        d["key"]

    d["key"] = "value"
    assert d["key"] == "value"
    assert len(d) == 1
    assert "key" in d
    assert list(d) == ["key"]
    del d["key"]
    assert not d
    assert "key" not in d
    with pytest.raises(KeyError):
        del d["key"]

    d["subdict"] = {"key": "value"}
    sd = d["subdict"]
    assert isinstance(sd, FoamDictionary)
    assert sd["key"] == "value"
    assert len(sd) == 1
    assert list(sd) == ["key"]

    d["subdict2"] = d["subdict"]
    sd2 = d["subdict2"]
    assert isinstance(sd2, FoamDictionary)
    assert sd2["key"] == "value"
    assert len(sd) == 1
    assert list(sd) == ["key"]

    sd["subsubdict"] = d["subdict"]
    ssd = sd["subsubdict"]
    assert isinstance(ssd, FoamDictionary)
    assert ssd["key"] == "value"

    sd["list"] = [1, 2, 3]
    assert sd["list"] == [1, 2, 3]

    sd["nestedList"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sd["nestedList"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
