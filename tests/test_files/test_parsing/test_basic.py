import numpy as np
import pytest
from foamlib import Dimensioned, DimensionSet, FoamFileDecodeError
from foamlib._files._parsing import ParsedFile


def test_parse_value() -> None:
    assert ParsedFile(b"1")[()] == 1
    assert ParsedFile(b"1.")[()] == 1.0
    assert ParsedFile(b"1.1")[()] == 1.1
    assert ParsedFile(b".1")[()] == 0.1
    assert ParsedFile(b"1.0e-3")[()] == 1.0e-3
    assert ParsedFile(b"1e-3")[()] == 1e-3
    assert ParsedFile(b"yes")[()] is True
    assert ParsedFile(b"no")[()] is False
    assert ParsedFile(b"word")[()] == "word"
    assert ParsedFile(b"inference")[()] == "inference"
    assert ParsedFile(b"word word")[()] == ("word", "word")
    assert ParsedFile(b'"a string"')[()] == '"a string"'
    assert ParsedFile(b"uniform 1")[()] == 1
    assert ParsedFile(b"uniform 1.0")[()] == 1.0
    assert ParsedFile(b"uniform 1.0e-3")[()] == 1.0e-3
    assert ParsedFile(b"(word word)")[()] == ["word", "word"]
    lst = ParsedFile(b"(1 2 3)")[()]
    assert isinstance(lst, np.ndarray)
    assert np.array_equal(lst, [1, 2, 3])
    lst = ParsedFile(b"list (1.0 2 3);")[("list",)]
    assert isinstance(lst, list)
    assert lst == [1.0, 2, 3]
    assert isinstance(lst[0], float)
    assert isinstance(lst[1], int)
    assert isinstance(lst[2], int)
    assert ParsedFile(b"()")[()] == []
    field = ParsedFile(b"uniform (1 2 3)")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2, 3])
    field = ParsedFile(b"nonuniform List<scalar> 2(1 2)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert np.array_equal(field, [1.0, 2.0])
    field = ParsedFile(b"nonuniform List<scalar> 2(1/*comment*/2)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert np.array_equal(field, [1.0, 2.0])
    field = ParsedFile(b"nonuniform List<scalar> 2{1}")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert np.array_equal(field, [1.0, 1.0])
    field = ParsedFile(b"nonuniform List<vector> 2((nan InFiNiTy -NaN)(1e3 .1 0.))")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field.shape == (2, 3)
    assert np.array_equal(field[0], [np.nan, np.inf, -np.nan], equal_nan=True)
    assert np.array_equal(field[1], [1e3, 0.1, 0.0])
    field = ParsedFile(b"nonuniform List<symmTensor> 0()")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field.shape == (0, 6)
    field = ParsedFile(b"nonuniform List<tensor> ()")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field.shape == (0, 9)
    arr = ParsedFile(b"3(1 2 3)")[()]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == int
    assert np.array_equal(arr, [1, 2, 3])
    arr = ParsedFile(b"3(1.0 2 3)")[()]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert np.array_equal(arr, [1.0, 2.0, 3.0])
    arr = ParsedFile(b"2((1 2 3) (4 5 6))")[()]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert np.array_equal(
        arr,
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    arr = ParsedFile(b"2((1\n2 3)\t(4 5 6))")[()]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert np.array_equal(
        arr,
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    lst = ParsedFile(b"2(3(1 2 3) 4(4 5 6 7))")[()]
    assert isinstance(lst, list)
    assert len(lst) == 2
    assert isinstance(lst[0], np.ndarray)
    assert np.array_equal(lst[0], [1, 2, 3])
    assert isinstance(lst[1], np.ndarray)
    assert np.array_equal(lst[1], [4, 5, 6, 7])
    lst = ParsedFile(b"2{(1 2 3)}")[()]
    assert isinstance(lst, np.ndarray)
    assert np.array_equal(
        lst,
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
    )
    assert ParsedFile(b"0()")[()] == []
    field = ParsedFile(b"nonuniform List<vector> 2((1 2 3) (4 5 6))")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(
        field,
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )
    field = ParsedFile(b"nonuniform List<vector> 2{(1 2 3)}")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(
        field,
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
    )
    field = ParsedFile(
        b"nonuniform List<scalar> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@)"
    )[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2])
    field = ParsedFile(
        b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [[1, 2, 3], [4, 5, 6]])
    field = ParsedFile(b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)")[()]
    assert isinstance(field, np.ndarray)
    assert np.array_equal(field, [1, 2])
    assert ParsedFile(b"[1 1 -2 0 0 0 0]")[()] == DimensionSet(
        mass=1, length=1, time=-2
    )
    dimensioned = ParsedFile(b"g [1 1 -2 0 0 0 0] (0 0 -9.81)")[()]
    assert isinstance(dimensioned, Dimensioned)
    assert dimensioned.dimensions == DimensionSet(mass=1, length=1, time=-2)
    assert np.array_equal(dimensioned.value, [0, 0, -9.81])
    assert dimensioned.name == "g"
    dimensioned = ParsedFile(b"[1 1 -2 0 0 0 0] 9.81")[()]
    assert isinstance(dimensioned, Dimensioned)
    assert dimensioned.dimensions == DimensionSet(mass=1, length=1, time=-2)
    assert dimensioned.value == 9.81
    assert dimensioned.name is None
    lst = ParsedFile(b"blocks (hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1));")[
        ("blocks",)
    ]
    assert isinstance(lst, list)
    assert len(lst) == 5
    assert lst[0] == "hex"
    assert isinstance(lst[1], list)
    assert lst[1] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert isinstance(lst[2], list)  # ty: ignore[index-out-of-bounds]
    assert lst[2] == [1, 1, 1]  # ty: ignore[index-out-of-bounds]
    assert lst[3] == "simpleGrading"  # ty: ignore[index-out-of-bounds]
    assert isinstance(lst[4], list)  # ty: ignore[index-out-of-bounds]
    assert lst[4] == [1, 1, 1]  # ty: ignore[index-out-of-bounds]
    assert ParsedFile(b"(a b; c d;)")[()] == [("a", "b"), ("c", "d")]
    assert ParsedFile(b"(a {b c;} d {e g;})")[()] == [
        ("a", {"b": "c"}),
        ("d", {"e": "g"}),
    ]
    assert ParsedFile(b"(a (b c d); e {})")[()] == [("a", ["b", "c", "d"]), ("e", {})]
    assert ParsedFile(b"({a b; c d;} {e g;})")[()] == [{"a": "b", "c": "d"}, {"e": "g"}]
    assert ParsedFile(b"(water oil mercury air)")[()] == [
        "water",
        "oil",
        "mercury",
        "air",
    ]
    assert ParsedFile(b"div(phi,U)")[()] == "div(phi,U)"
    assert ParsedFile(b"U.component(1)")[()] == "U.component(1)"
    assert ParsedFile(b"div(nuEff*dev(T(grad(U))))")[()] == "div(nuEff*dev(T(grad(U))))"
    assert (
        ParsedFile(b"div((nuEff*dev(T(grad(U)))))")[()]
        == "div((nuEff*dev(T(grad(U)))))"
    )
    assert ParsedFile(b"((air and water) { type constant; sigma 0.07; })")[()] == [
        (["air", "and", "water"], {"type": "constant", "sigma": 0.07})
    ]
    assert ParsedFile(b"[]")[()] == DimensionSet()
    assert ParsedFile(b"object f.1;")[("object",)] == "f.1"
    assert ParsedFile(b"keyword;")[("keyword",)] is None


def test_parse_directive() -> None:
    assert ParsedFile(b'#include "filename"')[("#include",)] == '"filename"'
    assert (
        ParsedFile(b"functions\n{\n#includeFunc funcName\nsubdict{}}")[
            ("functions", "#includeFunc")
        ]
        == "funcName"
    )


def test_parse_invalid_content() -> None:
    """Test that FoamFileDecodeError is raised for malformed content that cannot be parsed."""
    # Test malformed syntax that will cause the parser to fail
    with pytest.raises(FoamFileDecodeError):
        ParsedFile(b"key value; unclosed {")

    with pytest.raises(FoamFileDecodeError):
        ParsedFile(b"key { value; } extra }")

    with pytest.raises(FoamFileDecodeError):
        ParsedFile(b"{ orphaned brace")
