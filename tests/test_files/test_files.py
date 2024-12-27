import os
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator, Sequence
else:
    from typing import Generator, Sequence

import numpy as np
import pytest
from foamlib import FoamCase, FoamFieldFile, FoamFile


def test_write_read(tmp_path: Path) -> None:
    path = tmp_path / "testDict"
    d = FoamFile(path)
    assert d.path == path
    with pytest.raises(FileNotFoundError):
        d["key"]

    with d, pytest.raises(FileNotFoundError):
        d["key"]

    d[None] = "touch"
    assert len(d) == 1
    assert d[None] == "touch"
    assert list(d) == [None]
    del d[None]

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
    assert "FoamFile" in d
    del d["key"]
    assert not d
    assert "key" not in d
    with pytest.raises(KeyError):
        del d["key"]

    assert d.version == 2.0
    assert d.format == "ascii"
    assert d.class_ == "dictionary"
    assert d.location == f'"{d.path.parent.name}"'
    assert d.object_ == d.path.name

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


def test_new_field(tmp_path: Path) -> None:
    Path(tmp_path / "testField").touch()
    f = FoamFieldFile(tmp_path / "testField")
    f.internal_field = [1, 2, 3]
    field = f.internal_field
    assert isinstance(field, np.ndarray)
    assert np.array_equal(f.internal_field, [1, 2, 3])
    assert f.class_ == "volVectorField"


@pytest.fixture
def cavity() -> Generator[FoamCase, None, None]:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "incompressible" / "icoFoam" / "cavity" / "cavity"
    of11_path = tutorials_path / "incompressibleFluid" / "cavity"

    case = FoamCase(path if path.exists() else of11_path)

    with case.clone() as clone:
        yield clone


def test_dimensions(cavity: FoamCase) -> None:
    assert cavity[0]["p"].dimensions == FoamFile.DimensionSet(length=2, time=-2)
    assert cavity[0]["U"].dimensions == FoamFile.DimensionSet(length=1, time=-1)

    cavity[0]["p"].dimensions = FoamFile.DimensionSet(mass=1, length=1, time=-2)

    assert cavity[0]["p"].dimensions == FoamFile.DimensionSet(mass=1, length=1, time=-2)


def test_boundary_field(cavity: FoamCase) -> None:
    moving_wall = cavity[0]["p"].boundary_field["movingWall"]
    assert isinstance(moving_wall, FoamFieldFile.BoundarySubDict)
    assert moving_wall.type == "zeroGradient"
    assert "value" not in moving_wall

    moving_wall.type = "fixedValue"
    moving_wall.value = 0

    assert moving_wall.type == "fixedValue"
    assert moving_wall.value == 0


def test_mesh(cavity: FoamCase) -> None:
    cavity.run(parallel=False)

    file = cavity.file("constant/polyMesh/points")

    assert None in file
    assert None in list(file)

    points = file[None]

    assert isinstance(points, Sequence)
    assert isinstance(points[0], Sequence)
    assert len(points[0]) == 3


def test_internal_field(cavity: FoamCase) -> None:
    blocks = cavity.block_mesh_dict["blocks"]
    assert isinstance(blocks, list)
    sizes = blocks[2]
    assert isinstance(sizes, list)
    size = np.prod(sizes)

    p_arr = np.zeros(size)
    U_arr = np.zeros((size, 3))

    cavity[0]["p"].internal_field = p_arr  # type: ignore [assignment]
    cavity[0]["U"].internal_field = U_arr  # type: ignore [assignment]

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    p_arr = np.arange(size) * 1e-6  # type: ignore [assignment]
    U_arr = np.full((size, 3), [-1e-6, 1e-6, 0]) * np.arange(size)[:, np.newaxis]

    cavity[0]["p"].internal_field = p_arr  # type: ignore [assignment]
    cavity[0]["U"].internal_field = U_arr  # type: ignore [assignment]

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    cavity.run(parallel=False)


def test_fv_schemes(cavity: FoamCase) -> None:
    div_schemes = cavity.fv_schemes["divSchemes"]
    assert isinstance(div_schemes, FoamFile.SubDict)
    scheme = div_schemes["div(phi,U)"]
    assert isinstance(scheme, tuple)
    assert len(scheme) >= 2
    assert scheme[0] == "Gauss"


def test_binary_field(cavity: FoamCase) -> None:
    cavity.control_dict["writeFormat"] = "binary"

    cavity.run(parallel=False)

    p_bin = cavity[-1]["p"].internal_field
    assert isinstance(p_bin, np.ndarray)
    U_bin = cavity[-1]["U"].internal_field
    assert isinstance(U_bin, np.ndarray)
    assert U_bin.shape == (len(p_bin), 3)

    cavity.clean()

    p_arr = np.arange(len(p_bin)) * 1e-6
    U_arr = np.full_like(U_bin, [-1e-6, 1e-6, 0]) * np.arange(len(U_bin))[:, np.newaxis]

    cavity[0]["p"].internal_field = p_arr  # type: ignore [assignment]
    cavity[0]["U"].internal_field = U_arr

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    cavity.run(parallel=False)


def test_compressed_field(cavity: FoamCase) -> None:
    cavity.control_dict["writeCompression"] = True

    cavity.run(parallel=False)

    p_bin = cavity[-1]["p"].internal_field
    assert isinstance(p_bin, np.ndarray)
    U_bin = cavity[-1]["U"].internal_field
    assert isinstance(U_bin, np.ndarray)
    assert U_bin.shape == (len(p_bin), 3)

    cavity.clean()

    p_arr = np.arange(len(p_bin)) * 1e-6
    U_arr = np.full_like(U_bin, [-1e-6, 1e-6, 0]) * np.arange(len(U_bin))[:, np.newaxis]

    cavity[0]["p"].internal_field = p_arr  # type: ignore [assignment]
    cavity[0]["U"].internal_field = U_arr

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    cavity.run(parallel=False)
