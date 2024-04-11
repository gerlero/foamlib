from foamlib import FoamFile
from foamlib._dictionaries._serialization import _serialize_value


def test_serialize_value() -> None:
    assert _serialize_value(1) == "1"
    assert _serialize_value(1.0) == "1.0"
    assert _serialize_value(1.0e-3) == "0.001"
    assert _serialize_value(True) == "yes"
    assert _serialize_value(False) == "no"
    assert _serialize_value("word") == "word"
    assert _serialize_value("word word") == "word word"
    assert _serialize_value('"a string"') == '"a string"'
    assert _serialize_value(1, assume_field=True) == "uniform 1"
    assert _serialize_value(1.0, assume_field=True) == "uniform 1.0"
    assert _serialize_value(1.0e-3, assume_field=True) == "uniform 0.001"
    assert _serialize_value([1.0, 2.0, 3.0]) == "(1.0 2.0 3.0)"
    assert _serialize_value([1, 2, 3], assume_field=True) == "uniform (1 2 3)"
    assert (
        _serialize_value([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], assume_field=True)
        == "nonuniform List<scalar> 10(1 2 3 4 5 6 7 8 9 10)"
    )
    assert (
        _serialize_value([[1, 2, 3], [4, 5, 6]], assume_field=True)
        == "nonuniform List<vector> 2((1 2 3) (4 5 6))"
    )
    assert (
        _serialize_value(FoamFile.DimensionSet(mass=1, length=1, time=-2))
        == "[1 1 -2 0 0 0 0]"
    )
    assert (
        _serialize_value(
            FoamFile.Dimensioned(
                name="g",
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            )
        )
        == "g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        _serialize_value(
            FoamFile.Dimensioned(
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
            )
        )
        == "[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        _serialize_value("hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)")
        == "hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
