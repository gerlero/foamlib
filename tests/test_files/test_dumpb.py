from foamlib import FoamFile
from foamlib._files._serialization import Kind, dumpb


def test_serialize_data() -> None:
    assert dumpb(1) == b"1"
    assert dumpb(1.0) == b"1.0"
    assert dumpb(1.0e-3) == b"0.001"
    assert dumpb(True) == b"yes"
    assert dumpb(False) == b"no"
    assert dumpb("word") == b"word"
    assert dumpb(("word", "word")) == b"word word"
    assert dumpb('"a string"') == b'"a string"'
    assert dumpb(1, kind=Kind.FIELD) == b"uniform 1"
    assert dumpb(1.0, kind=Kind.FIELD) == b"uniform 1.0"
    assert dumpb(1.0e-3, kind=Kind.FIELD) == b"uniform 0.001"
    assert dumpb([1.0, 2.0, 3.0]) == b"(1.0 2.0 3.0)"
    assert dumpb([1, 2, 3], kind=Kind.FIELD) == b"uniform (1 2 3)"
    assert (
        dumpb([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kind=Kind.FIELD)
        == b"nonuniform List<scalar> 10(1 2 3 4 5 6 7 8 9 10)"
    )
    assert (
        dumpb([[1, 2, 3], [4, 5, 6]], kind=Kind.FIELD)
        == b"nonuniform List<vector> 2((1 2 3) (4 5 6))"
    )
    assert dumpb(1, kind=Kind.BINARY_FIELD) == b"uniform 1"
    assert dumpb(1.0, kind=Kind.BINARY_FIELD) == b"uniform 1.0"
    assert dumpb([1, 2, 3], kind=Kind.BINARY_FIELD) == b"uniform (1 2 3)"
    assert (
        dumpb([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kind=Kind.BINARY_FIELD)
        == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
    )
    assert (
        dumpb([[1, 2, 3], [4, 5, 6]], kind=Kind.BINARY_FIELD)
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
        dumpb(("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]))
        == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert dumpb([{"a": "b"}, {"c": "d"}]) == b"(a b; c d;)"
    assert dumpb([{"a": {"b": "c"}}, {"d": {"e": "g"}}]) == b"(a {b c;} d {e g;})"
    assert dumpb([{"a": [0, 1, 2]}, {"b": {}}]) == b"(a (0 1 2); b {})"
    assert dumpb(["water", "oil", "mercury", "air"]) == b"(water oil mercury air)"
    assert dumpb("div(phi,U)") == b"div(phi,U)"
