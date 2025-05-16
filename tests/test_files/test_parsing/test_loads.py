import numpy as np
from foamlib import FoamFile


def test_loads() -> None:
    assert FoamFile.loads("") == {}
    assert FoamFile.loads("1") == 1
    assert FoamFile.loads("FoamFile {} 1") == 1
    assert FoamFile.loads("FoamFile {} 1", include_header=True) == {
        "FoamFile": {},
        None: 1,
    }
    assert FoamFile.loads("1.0") == 1.0
    assert FoamFile.loads("1.0e-3") == 1.0e-3
    assert FoamFile.loads("yes") is True
    assert FoamFile.loads("no") is False
    assert FoamFile.loads("word") == "word"
    assert FoamFile.loads("word word") == ("word", "word")
    assert FoamFile.loads('"a string"') == '"a string"'
    assert FoamFile.loads("(word word)") == ["word", "word"]
    assert FoamFile.loads("keyword value;") == {"keyword": "value"}
    assert FoamFile.loads("uniform 1") == 1
    assert FoamFile.loads("uniform 1.0") == 1.0
    assert FoamFile.loads("uniform 1.0e-3") == 1.0e-3
    assert np.array_equal(FoamFile.loads("(1 2 3)"), [1, 2, 3])  # type: ignore[arg-type]
    assert np.array_equal(FoamFile.loads("3(1 2 3)"), [1, 2, 3])  # type: ignore[arg-type]
    assert np.array_equal(
        FoamFile.loads("2((1 2 3) (4 5 6))"),  # type: ignore[arg-type]
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )
    assert np.array_equal(
        FoamFile.loads("2{(1 2 3)}"),  # type: ignore[arg-type]
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
    )
    assert FoamFile.loads("0()") == []
    assert np.array_equal(
        FoamFile.loads(
            b"nonuniform List<scalar> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@)"
        ),  # type: ignore[arg-type]
        [1, 2],
    )
    assert np.array_equal(
        FoamFile.loads(
            b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
        ),  # type: ignore[arg-type]
        [[1, 2, 3], [4, 5, 6]],
    )
    assert np.array_equal(
        FoamFile.loads(b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)"),  # type: ignore[arg-type]
        [1, 2],
    )
    assert FoamFile.loads("[1 1 -2 0 0 0 0]") == FoamFile.DimensionSet(
        mass=1, length=1, time=-2
    )
    dimensioned = FoamFile.loads("g [1 1 -2 0 0 0 0] (0 0 -9.81)")
    assert isinstance(dimensioned, FoamFile.Dimensioned)
    assert dimensioned.dimensions == FoamFile.DimensionSet(mass=1, length=1, time=-2)
    assert np.array_equal(dimensioned.value, [0, 0, -9.81])
    assert dimensioned.name == "g"
    dimensioned = FoamFile.loads("[1 1 -2 0 0 0 0] 9.81")
    assert isinstance(dimensioned, FoamFile.Dimensioned)
    assert dimensioned.dimensions == FoamFile.DimensionSet(mass=1, length=1, time=-2)
    assert dimensioned.value == 9.81
    assert dimensioned.name is None
    assert FoamFile.loads("a {b c; d e;}") == {"a": {"b": "c", "d": "e"}}
    assert FoamFile.loads("(a b; c d;)") == [("a", "b"), ("c", "d")]
    assert FoamFile.loads("keyword;") == {"keyword": ""}
    assert FoamFile.loads("#include $FOAM_CASE/simControls") == {
        "#include": "$FOAM_CASE/simControls"
    }
    faces = FoamFile.loads("2(3(1 2 3) 4(4 5 6 7))")
    assert isinstance(faces, list)
    assert len(faces) == 2
    assert isinstance(faces[0], np.ndarray)
    assert faces[0].shape == (3,)
    assert faces[0].dtype == np.int64
    assert isinstance(faces[1], np.ndarray)
    assert faces[1].shape == (4,)
    assert faces[1].dtype == np.int64
    assert np.array_equal(faces[0], [1, 2, 3])
    assert np.array_equal(faces[1], [4, 5, 6, 7])
    faces = FoamFile.loads(
        b"3\n(\x00\x00\x00\x00\x03\x00\x00\x00\x07\x00\x00\x00)\n7\n(h\xcb\r\x00\x1f\xcd\r\x00\x08\x1b\x0e\x00\xd7\xa85\x00\xda\xa85\x00\xd9\xa85\x00\xd8\xa85\x00)"
    )
    assert isinstance(faces, tuple)
    assert len(faces) == 2
    assert isinstance(faces[0], np.ndarray)
    assert faces[0].shape == (3,)
    assert faces[0].dtype == np.int32
    assert isinstance(faces[1], np.ndarray)
    assert faces[1].shape == (7,)
    assert faces[1].dtype == np.int32
    assert np.array_equal(faces[1][faces[0][0] : faces[0][1]], [904040, 904479, 924424])
    assert np.array_equal(
        faces[1][faces[0][1] : faces[0][2]], [3516631, 3516634, 3516633, 3516632]
    )
