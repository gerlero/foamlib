from foamlib import FoamFile
from foamlib._dictionaries._parsing import _VALUE


def test_parse_value() -> None:
    assert _VALUE.parse_string("1")[0] == 1
    assert _VALUE.parse_string("1.0")[0] == 1.0
    assert _VALUE.parse_string("1.0e-3")[0] == 1.0e-3
    assert _VALUE.parse_string("yes")[0] is True
    assert _VALUE.parse_string("no")[0] is False
    assert _VALUE.parse_string("word")[0] == "word"
    assert _VALUE.parse_string("word word")[0] == "word word"
    assert _VALUE.parse_string('"a string"')[0] == '"a string"'
    assert _VALUE.parse_string("uniform 1")[0] == 1
    assert _VALUE.parse_string("uniform 1.0")[0] == 1.0
    assert _VALUE.parse_string("uniform 1.0e-3")[0] == 1.0e-3
    assert _VALUE.parse_string("(1.0 2.0 3.0)")[0] == [1.0, 2.0, 3.0]
    assert _VALUE.parse_string("uniform (1 2 3)")[0] == [1, 2, 3]
    assert _VALUE.parse_string("nonuniform List<scalar> 2(1 2)")[0] == [1, 2]
    assert _VALUE.parse_string("nonuniform List<scalar> 2{1}")[0] == [1, 1]
    assert _VALUE.parse_string("3(1 2 3)")[0] == [1, 2, 3]
    assert _VALUE.parse_string("2((1 2 3) (4 5 6))")[0] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert _VALUE.parse_string("2{(1 2 3)}")[0] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert _VALUE.parse_string("nonuniform List<vector> 2((1 2 3) (4 5 6))")[0] == [
        [1, 2, 3],
        [4, 5, 6],
    ]
    assert _VALUE.parse_string("nonuniform List<vector> 2{(1 2 3)}")[0] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert _VALUE.parse_string("[1 1 -2 0 0 0 0]")[0] == FoamFile.DimensionSet(
        mass=1, length=1, time=-2
    )
    assert _VALUE.parse_string("g [1 1 -2 0 0 0 0] (0 0 -9.81)")[
        0
    ] == FoamFile.Dimensioned(
        name="g",
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
        value=[0, 0, -9.81],
    )
    assert _VALUE.parse_string("[1 1 -2 0 0 0 0] 9.81")[0] == FoamFile.Dimensioned(
        dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
    )
    assert (
        _VALUE.parse_string("hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)")[0]
        == "hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
