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
