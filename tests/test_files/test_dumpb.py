from foamlib import FoamFile
from foamlib._files._serialization import dumpb


def test_serialize_data() -> None:
    assert dumpb(1) == b"1"
    assert dumpb(1.0) == b"1.0"
    assert dumpb(1.0e-3) == b"0.001"
    assert dumpb(True) == b"yes"
    assert dumpb(False) == b"no"
    assert dumpb("word") == b"word"
    assert dumpb(("word", "word"), assume_data_entries=True) == b"word word"
    assert dumpb('"a string"') == b'"a string"'
    assert dumpb(1, assume_field=True) == b"uniform 1"
    assert dumpb(1.0, assume_field=True) == b"uniform 1.0"
    assert dumpb(1.0e-3, assume_field=True) == b"uniform 0.001"
    assert dumpb([1.0, 2.0, 3.0]) == b"(1.0 2.0 3.0)"
    assert dumpb([1, 2, 3], assume_field=True) == b"uniform (1 2 3)"
    assert (
        dumpb([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], assume_field=True)
        == b"nonuniform List<scalar> 10(1 2 3 4 5 6 7 8 9 10)"
    )
    assert (
        dumpb([[1, 2, 3], [4, 5, 6]], assume_field=True)
        == b"nonuniform List<vector> 2((1 2 3) (4 5 6))"
    )
    assert (
        dumpb([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], assume_field=True, binary_fields=True)
        == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
    )
    assert (
        dumpb([[1, 2, 3], [4, 5, 6]], assume_field=True, binary_fields=True)
        == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )
    assert (
        dumpb(FoamFile.DimensionSet(mass=1, length=1, time=-2)) == b"[1 1 -2 0 0 0 0]"
    )
    assert (
        dumpb(
            FoamFile.Dimensioned(
                name="g",
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            )
        )
        == b"g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumpb(
            FoamFile.Dimensioned(
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
            )
        )
        == b"[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumpb(
            ("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]),
            assume_data_entries=True,
        )
        == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert dumpb([{"a": "b"}, {"c": "d"}]) == b"(a b; c d;)"
    assert (
        dumpb([{"a": {"b": "c"}}, {"d": {"e": "g"}}])
        == b"(a\n{\nb c;\n} d\n{\ne g;\n})"
    )
    assert dumpb([{"a": [0, 1, 2]}, {"b": {}}]) == b"(a (0 1 2); b\n{\n\n})"
