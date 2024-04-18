from foamlib import FoamFile
from foamlib._files._serialization import dumps


def test_serialize_data() -> None:
    assert dumps(1) == "1"
    assert dumps(1.0) == "1.0"
    assert dumps(1.0e-3) == "0.001"
    assert dumps(True) == "yes"
    assert dumps(False) == "no"
    assert dumps("word") == "word"
    assert dumps(("word", "word"), assume_data_entries=True) == "word word"
    assert dumps('"a string"') == '"a string"'
    assert dumps(1, assume_field=True) == "uniform 1"
    assert dumps(1.0, assume_field=True) == "uniform 1.0"
    assert dumps(1.0e-3, assume_field=True) == "uniform 0.001"
    assert dumps([1.0, 2.0, 3.0]) == "(1.0 2.0 3.0)"
    assert dumps([1, 2, 3], assume_field=True) == "uniform (1 2 3)"
    assert (
        dumps([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], assume_field=True)
        == "nonuniform List<scalar> 10(1 2 3 4 5 6 7 8 9 10)"
    )
    assert (
        dumps([[1, 2, 3], [4, 5, 6]], assume_field=True)
        == "nonuniform List<vector> 2((1 2 3) (4 5 6))"
    )
    assert dumps(FoamFile.DimensionSet(mass=1, length=1, time=-2)) == "[1 1 -2 0 0 0 0]"
    assert (
        dumps(
            FoamFile.Dimensioned(
                name="g",
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            )
        )
        == "g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(
            FoamFile.Dimensioned(
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
            )
        )
        == "[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(
            ("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]),
            assume_data_entries=True,
        )
        == "hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert dumps([{"a": "b"}, {"c": "d"}]) == "(a b; c d;)"
    assert (
        dumps([{"a": {"b": "c"}}, {"d": {"e": "g"}}]) == "(a\n{\nb c;\n} d\n{\ne g;\n})"
    )
    assert dumps([{"a": [0, 1, 2]}, {"b": {}}]) == "(a (0 1 2); b\n{\n\n})"
