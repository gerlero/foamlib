import os
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

import numpy as np
import pytest
from foamlib import FoamCase, FoamFieldFile, FoamFile


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
    assert isinstance(sd, FoamFile.SubDict)
    assert sd["key"] == "value"
    assert len(sd) == 1
    assert list(sd) == ["key"]

    d["subdict2"] = d["subdict"]
    sd2 = d["subdict2"]
    assert isinstance(sd2, FoamFile.SubDict)
    assert sd2["key"] == "value"
    assert len(sd) == 1
    assert list(sd) == ["key"]

    sd["subsubdict"] = d["subdict"]
    ssd = sd["subsubdict"]
    assert isinstance(ssd, FoamFile.SubDict)
    assert ssd["key"] == "value"

    sd["list"] = [1, 2, 3]
    assert sd["list"] == [1, 2, 3]

    sd["nestedList"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sd["nestedList"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    sd["g"] = FoamFile.Dimensioned(
        name="g", dimensions=[1, 1, -2, 0, 0, 0, 0], value=[0, 0, -9.81]
    )
    assert sd["g"] == FoamFile.Dimensioned(
        name="g",
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
        value=[0, 0, -9.81],
    )

    with d:
        lst = d["subdict", "list"]
        assert isinstance(lst, list)
        lst[0] = 0
        assert lst == [0, 2, 3]
        assert d["subdict", "list"] == [1, 2, 3]


PITZ = FoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "incompressible" / "simpleFoam" / "pitzDaily"
)


@pytest.fixture
def pitz(tmp_path: Path) -> FoamCase:
    return PITZ.clone(tmp_path / PITZ.name)


def test_dimensions(pitz: FoamCase) -> None:
    assert pitz[0]["p"].dimensions == FoamFile.DimensionSet(length=2, time=-2)
    assert pitz[0]["U"].dimensions == FoamFile.DimensionSet(length=1, time=-1)

    pitz[0]["p"].dimensions = FoamFile.DimensionSet(mass=1, length=1, time=-2)

    assert pitz[0]["p"].dimensions == FoamFile.DimensionSet(mass=1, length=1, time=-2)


def test_boundary_field(pitz: FoamCase) -> None:
    outlet = pitz[0]["p"].boundary_field["outlet"]
    assert isinstance(outlet, FoamFieldFile.BoundarySubDict)
    assert outlet.type == "fixedValue"
    assert outlet.value == 0

    outlet.type = "zeroGradient"
    del outlet.value

    assert outlet.type == "zeroGradient"
    assert "value" not in outlet


def test_internal_field(pitz: FoamCase) -> None:
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

    p_arr = np.zeros(size)
    U_arr = np.zeros((size, 3))

    pitz[0]["p"].internal_field = p_arr
    pitz[0]["U"].internal_field = U_arr

    assert pitz[0]["p"].internal_field == pytest.approx(p_arr)
    U = pitz[0]["U"].internal_field
    assert isinstance(U, Sequence)
    for u, u_arr in zip(U, U_arr):
        assert u == pytest.approx(u_arr)

    p_arr = np.arange(size) * 1e-6
    U_arr = np.full((size, 3), [-1e-6, 1e-6, 0]) * np.arange(size)[:, np.newaxis]

    pitz[0]["p"].internal_field = p_arr
    pitz[0]["U"].internal_field = U_arr

    assert pitz[0]["p"].internal_field == pytest.approx(p_arr)
    U = pitz[0]["U"].internal_field
    assert isinstance(U, Sequence)
    for u, u_arr in zip(U, U_arr):
        assert u == pytest.approx(u_arr)

    pitz.run()


def test_fv_schemes(pitz: FoamCase) -> None:
    div_schemes = pitz.fv_schemes["divSchemes"]
    assert isinstance(div_schemes, FoamFile.SubDict)
    scheme = div_schemes["div(phi,U)"]
    assert isinstance(scheme, tuple)
    assert scheme == ("bounded", "Gauss", "linearUpwind", "grad(U)")


def test_binary_field(pitz: FoamCase) -> None:
    pitz.control_dict["writeFormat"] = "binary"

    pitz.run()

    p_bin = pitz[-1]["p"].internal_field
    assert isinstance(p_bin, Sequence)
    U_bin = pitz[-1]["U"].internal_field
    assert isinstance(U_bin, Sequence)
    assert isinstance(U_bin[0], Sequence)
    assert len(U_bin[0]) == 3
    size = len(p_bin)
    assert len(U_bin) == size

    p_arr = np.arange(size) * 1e-6
    U_arr = np.full((size, 3), [-1e-6, 1e-6, 0]) * np.arange(size)[:, np.newaxis]

    pitz[0]["p"].internal_field = p_arr
    pitz[0]["U"].internal_field = U_arr

    assert pitz[0]["p"].internal_field == pytest.approx(p_arr)
    U = pitz[0]["U"].internal_field
    assert isinstance(U, Sequence)
    for u, u_arr in zip(U, U_arr):
        assert u == pytest.approx(u_arr)

    pitz.run()
