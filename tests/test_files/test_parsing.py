import numpy as np
from foamlib import FoamFile
from foamlib._files._parsing import Parsed


def test_parse_value() -> None:
    assert Parsed(b"1")[()] == 1
    assert Parsed(b"1.0")[()] == 1.0
    assert Parsed(b"1.0e-3")[()] == 1.0e-3
    assert Parsed(b"yes")[()] is True
    assert Parsed(b"no")[()] is False
    assert Parsed(b"word")[()] == "word"
    assert Parsed(b"word word")[()] == ("word", "word")
    assert Parsed(b'"a string"')[()] == '"a string"'
    assert Parsed(b"uniform 1")[()] == 1
    assert Parsed(b"uniform 1.0")[()] == 1.0
    assert Parsed(b"uniform 1.0e-3")[()] == 1.0e-3
    assert Parsed(b"(1.0 2.0 3.0)")[()] == [1.0, 2.0, 3.0]
    field = Parsed(b"uniform (1 2 3)")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2, 3])
    field = Parsed(b"nonuniform List<scalar> 2(1 2)")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2])
    field = Parsed(b"nonuniform List<scalar> 2{1}")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 1])
    assert Parsed(b"3(1 2 3)")[()] == [1, 2, 3]
    assert Parsed(b"2((1 2 3) (4 5 6))")[()] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert Parsed(b"2{(1 2 3)}")[()] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    field = Parsed(b"nonuniform List<vector> 2((1 2 3) (4 5 6))")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(
        field,
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )
    field = Parsed(b"nonuniform List<vector> 2{(1 2 3)}")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(
        field,
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
    )
    field = Parsed(
        b"nonuniform List<scalar> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@)"
    )[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2])
    field = Parsed(
        b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [[1, 2, 3], [4, 5, 6]])
    field = Parsed(b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2])
    assert Parsed(b"[1 1 -2 0 0 0 0]")[()] == FoamFile.DimensionSet(
        mass=1, length=1, time=-2
    )
    assert Parsed(b"g [1 1 -2 0 0 0 0] (0 0 -9.81)")[()] == FoamFile.Dimensioned(
        name="g",
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
        value=[0, 0, -9.81],
    )
    assert Parsed(b"[1 1 -2 0 0 0 0] 9.81")[()] == FoamFile.Dimensioned(
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
    )
    assert Parsed(b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)")[()] == (
        "hex",
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 1, 1],
        "simpleGrading",
        [1, 1, 1],
    )
    assert Parsed(b"(a b; c d;)")[()] == [("a", "b"), ("c", "d")]
    assert Parsed(b"(a {b c;} d {e g;})")[()] == [
        ("a", {"b": "c"}),
        ("d", {"e": "g"}),
    ]
    assert Parsed(b"(a (0 1 2); b {})")[()] == [("a", [0, 1, 2]), ("b", {})]
    assert Parsed(b"({a b; c d;} {e g;})")[()] == [{"a": "b", "c": "d"}, {"e": "g"}]
    assert Parsed(b"(water oil mercury air)")[()] == ["water", "oil", "mercury", "air"]
    assert Parsed(b"div(phi,U)")[()] == "div(phi,U)"
    assert Parsed(b"div(nuEff*dev(T(grad(U))))")[()] == "div(nuEff*dev(T(grad(U))))"
    assert Parsed(b"((air and water) { type constant; sigma 0.07; })")[()] == [
        (["air", "and", "water"], {"type": "constant", "sigma": 0.07})
    ]
    assert Parsed(b"[]")[()] == FoamFile.DimensionSet()
    assert Parsed(b"object f.1;")[("object",)] == "f.1"
