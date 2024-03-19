import pytest

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from foamlib import FoamFile, FoamDictionary, FoamCase


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


PITZ = FoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "incompressible" / "simpleFoam" / "pitzDaily"
)


@pytest.fixture
def pitz(tmp_path: Path) -> FoamCase:
    return PITZ.clone(tmp_path / PITZ.name)


def test_field(pitz: FoamCase) -> None:
    pitz[0]["p"].internal_field = 0.5
    pitz[0]["U"].internal_field = [1.5, 2.0, 3]

    assert pitz[0]["p"].internal_field == 0.5
    assert pitz[0]["U"].internal_field == [1.5, 2.0, 3]

    pitz.run()

    p = pitz[-1]["p"].internal_field
    assert isinstance(p, Sequence)
    U = pitz[-1]["U"].internal_field
    assert isinstance(U, Sequence)
    size = len(p)
    assert len(U) == size

    pitz.clean()

    p_arr = np.arange(size) * 1e-6
    U_arr = np.full((size, 3), [-1e-6, 1e-6, 0]) * np.arange(size)[:, np.newaxis]

    pitz[0]["p"].internal_field = p_arr  # type: ignore
    pitz[0]["U"].internal_field = U_arr

    assert pitz[0]["p"].internal_field == pytest.approx(p_arr)
    U = pitz[-1]["U"].internal_field
    assert isinstance(U, Sequence)
    for u, u_arr in zip(U, U_arr):
        assert u == pytest.approx(u_arr)

    pitz.run()
