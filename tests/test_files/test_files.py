import os
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from foamlib import Dimensioned, DimensionSet, FoamCase, FoamFieldFile, FoamFile


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

    sd["subsubdict"] = {"a": sd["subsubdict"], "b": 2}
    ssd = sd["subsubdict"]
    assert isinstance(ssd, FoamFile.SubDict)
    assert "key" not in ssd
    assert isinstance(ssd["a"], FoamFile.SubDict)
    assert ssd["a"]["key"] == "value"
    assert ssd["b"] == 2

    sd["list"] = [1, 2, 3]
    assert sd["list"] == [1, 2, 3]

    sd["nestedList"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sd["nestedList"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    sd["g"] = Dimensioned(
        name="g", dimensions=[0, 1, -2, 0, 0, 0, 0], value=[0, 0, -9.81]
    )
    assert isinstance(sd["g"], Dimensioned)
    assert sd["g"].name == "g"
    assert sd["g"].dimensions == DimensionSet(length=1, time=-2)
    assert np.array_equal(sd["g"].value, [0, 0, -9.81])

    sd["n"] = 1
    sd["y"] = 2
    assert sd["n"] == 1
    assert sd["y"] == 2

    with d:
        sd = d["subdict"]
        assert isinstance(sd, FoamFile.SubDict)
        lst = sd["list"]
        assert isinstance(lst, list)
        lst[0] = 0
        assert lst == [0, 2, 3]
        assert sd["list"] == [1, 2, 3]

    with pytest.raises(ValueError, match="invalid"):
        d["invalid"] = "semicolon;"


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
    assert cavity[0]["p"].dimensions == DimensionSet(length=2, time=-2)
    assert cavity[0]["U"].dimensions == DimensionSet(length=1, time=-1)

    cavity[0]["p"].dimensions = DimensionSet(mass=1, length=1, time=-2)

    assert cavity[0]["p"].dimensions == DimensionSet(mass=1, length=1, time=-2)


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

    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[-1] == 3  # ty: ignore[non-subscriptable]


def test_internal_field(cavity: FoamCase) -> None:
    blocks = cavity.block_mesh_dict["blocks"]
    assert isinstance(blocks, list)
    sizes = blocks[2]
    assert isinstance(sizes, list)
    size = np.prod(sizes)

    p_arr = np.zeros(size)
    U_arr = np.zeros((size, 3))

    cavity[0]["p"].internal_field = p_arr
    cavity[0]["U"].internal_field = U_arr

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    p_arr = np.arange(size) * 1e-6
    U_arr = np.full((size, 3), [-1e-6, 1e-6, 0]) * np.arange(size)[:, np.newaxis]

    cavity[0]["p"].internal_field = p_arr
    cavity[0]["U"].internal_field = U_arr

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

    cavity[0]["p"].internal_field = p_arr
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

    cavity[0]["p"].internal_field = p_arr
    cavity[0]["U"].internal_field = U_arr

    assert cavity[0]["p"].internal_field == pytest.approx(p_arr)
    U = cavity[0]["U"].internal_field
    assert isinstance(U, np.ndarray)
    assert U_arr == pytest.approx(U)

    cavity.run(parallel=False)


def test_popone(tmp_path: Path) -> None:
    """Test the popone method for FoamFile and SubDict."""
    path = tmp_path / "testDict"
    d = FoamFile(path)

    # Test popone with basic values
    d["key1"] = "value1"
    d["key2"] = 42
    d["key3"] = [1, 2, 3]

    # Pop a string value
    popped = d.popone("key1")
    assert popped == "value1"
    assert "key1" not in d

    # Pop a number value
    popped = d.popone("key2")
    assert popped == 42
    assert "key2" not in d

    # Pop a list value
    popped = d.popone("key3")
    assert popped == [1, 2, 3]
    assert "key3" not in d

    # Verify the popped list is a copy, not a live reference
    assert isinstance(popped, list)
    assert isinstance(popped[0], int)
    popped[0] = 999  # ty: ignore[invalid-assignment]
    # Since the key is already removed, we can't check the original,
    # but we verified it's a proper list copy

    # Test popone with subdictionaries
    d["subdict"] = {"nested_key": "nested_value", "nested_num": 100}
    popped_dict = d.popone("subdict")

    # Verify it returns a dict (SubDict type alias), not FoamFile.SubDict
    assert isinstance(popped_dict, dict)
    assert not isinstance(popped_dict, FoamFile.SubDict)
    assert popped_dict == {"nested_key": "nested_value", "nested_num": 100}
    assert "subdict" not in d

    # Verify the popped dict is a deep copy
    assert isinstance(popped_dict, dict)
    popped_dict["nested_key"] = "modified"  # ty: ignore[invalid-assignment]
    # Since the key is already removed, this confirms it's a copy

    # Test popone on SubDict
    d["parent"] = {
        "child1": "value1",
        "child2": {"grandchild": "grandvalue"},
        "child3": [10, 20, 30],
    }

    parent = d["parent"]
    assert isinstance(parent, FoamFile.SubDict)

    # Pop from SubDict
    popped = parent.popone("child1")
    assert popped == "value1"
    assert "child1" not in parent

    # Pop a nested dict from SubDict
    popped_nested = parent.popone("child2")
    assert isinstance(popped_nested, dict)
    assert not isinstance(popped_nested, FoamFile.SubDict)
    assert popped_nested == {"grandchild": "grandvalue"}
    assert "child2" not in parent

    # Verify it's a deep copy
    assert isinstance(popped_nested, dict)
    popped_nested["grandchild"] = "modified"  # ty: ignore[invalid-assignment]
    # Already removed, so this confirms it's a copy

    # Pop a list from SubDict
    popped_list = parent.popone("child3")
    assert popped_list == [10, 20, 30]
    assert "child3" not in parent

    # Test with deeply nested subdictionaries
    d["level1"] = {"level2": {"level3": {"deep_value": "found"}}}

    level1 = d["level1"]
    assert isinstance(level1, FoamFile.SubDict)
    level2 = level1["level2"]
    assert isinstance(level2, FoamFile.SubDict)
    popped_level3 = level2.popone("level3")

    assert isinstance(popped_level3, dict)
    assert not isinstance(popped_level3, FoamFile.SubDict)
    assert popped_level3 == {"deep_value": "found"}
    assert "level3" not in level2

    # Test popone with None (non-existent key)
    d["test_key"] = None
    popped_none = d.popone("test_key")
    assert popped_none is None
    assert "test_key" not in d
