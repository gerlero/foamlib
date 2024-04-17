from foamlib import FoamFile
from foamlib._files._serialization import serialize


def test_serialize_data() -> None:
    assert serialize(1) == "1"
    assert serialize(1.0) == "1.0"
    assert serialize(1.0e-3) == "0.001"
    assert serialize(True) == "yes"
    assert serialize(False) == "no"
    assert serialize("word") == "word"
    assert serialize(("word", "word"), assume_data_entries=True) == "word word"
    assert serialize('"a string"') == '"a string"'
    assert serialize(1, assume_field=True) == "uniform 1"
    assert serialize(1.0, assume_field=True) == "uniform 1.0"
    assert serialize(1.0e-3, assume_field=True) == "uniform 0.001"
    assert serialize([1.0, 2.0, 3.0]) == "(1.0 2.0 3.0)"
    assert serialize([1, 2, 3], assume_field=True) == "uniform (1 2 3)"
    assert (
        serialize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], assume_field=True)
        == "nonuniform List<scalar> 10(1 2 3 4 5 6 7 8 9 10)"
    )
    assert (
        serialize([[1, 2, 3], [4, 5, 6]], assume_field=True)
        == "nonuniform List<vector> 2((1 2 3) (4 5 6))"
    )
    assert (
        serialize(FoamFile.DimensionSet(mass=1, length=1, time=-2))
        == "[1 1 -2 0 0 0 0]"
    )
    assert (
        serialize(
            FoamFile.Dimensioned(
                name="g",
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            )
        )
        == "g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        serialize(
            FoamFile.Dimensioned(
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
            )
        )
        == "[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        serialize(
            ("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]),
            assume_data_entries=True,
        )
        == "hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert serialize([{"a": "b"}, {"c": "d"}]) == "(a b; c d;)"
    assert (
        serialize([{"a": {"b": "c"}}, {"d": {"e": "g"}}])
        == "(a\n{\nb c;\n} d\n{\ne g;\n})"
    )
    assert serialize([{"a": [0, 1, 2]}, {"b": {}}]) == "(a (0 1 2); b\n{\n\n})"
