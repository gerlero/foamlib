from foamlib import FoamFile
from foamlib._files._parsing import _DATA


def test_parse_value() -> None:
    assert _DATA.parse_string("1")[0] == 1
    assert _DATA.parse_string("1.0")[0] == 1.0
    assert _DATA.parse_string("1.0e-3")[0] == 1.0e-3
    assert _DATA.parse_string("yes")[0] is True
    assert _DATA.parse_string("no")[0] is False
    assert _DATA.parse_string("word")[0] == "word"
    assert _DATA.parse_string("word word")[0] == ("word", "word")
    assert _DATA.parse_string('"a string"')[0] == '"a string"'
    assert _DATA.parse_string("uniform 1")[0] == 1
    assert _DATA.parse_string("uniform 1.0")[0] == 1.0
    assert _DATA.parse_string("uniform 1.0e-3")[0] == 1.0e-3
    assert _DATA.parse_string("(1.0 2.0 3.0)")[0] == [1.0, 2.0, 3.0]
    assert _DATA.parse_string("uniform (1 2 3)")[0] == [1, 2, 3]
    assert _DATA.parse_string("nonuniform List<scalar> 2(1 2)")[0] == [1, 2]
    assert _DATA.parse_string("nonuniform List<scalar> 2{1}")[0] == [1, 1]
    assert _DATA.parse_string("3(1 2 3)")[0] == [1, 2, 3]
    assert _DATA.parse_string("2((1 2 3) (4 5 6))")[0] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert _DATA.parse_string("2{(1 2 3)}")[0] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert _DATA.parse_string("nonuniform List<vector> 2((1 2 3) (4 5 6))")[0] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert _DATA.parse_string("nonuniform List<vector> 2{(1 2 3)}")[0] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert _DATA.parse_string(
        "nonuniform List<scalar> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@)"
    )[0] == [1, 2]
    assert _DATA.parse_string(
        "nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )[0] == [[1, 2, 3], [4, 5, 6]]
    assert _DATA.parse_string("[1 1 -2 0 0 0 0]")[0] == FoamFile.DimensionSet(
        mass=1, length=1, time=-2
    )
    assert _DATA.parse_string("g [1 1 -2 0 0 0 0] (0 0 -9.81)")[
        0
    ] == FoamFile.Dimensioned(
        name="g",
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
        value=[0, 0, -9.81],
    )
    assert _DATA.parse_string("[1 1 -2 0 0 0 0] 9.81")[0] == FoamFile.Dimensioned(
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
    )
    assert _DATA.parse_string("hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)")[
        0
    ] == ("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1])
    assert _DATA.parse_string("(a b; c d;)")[0] == [{"a": "b"}, {"c": "d"}]
    assert _DATA.parse_string("(a {b c;} d {e g;})")[0] == [
        {"a": {"b": "c"}},
        {"d": {"e": "g"}},
    ]
    assert _DATA.parse_string("(a (0 1 2); b {})")[0] == [{"a": [0, 1, 2]}, {"b": {}}]
