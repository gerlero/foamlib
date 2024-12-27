import numpy as np
from foamlib import FoamFile
from foamlib._files._serialization import Kind, dumps


def test_serialize_data() -> None:
    assert dumps(1) == b"1"
    assert dumps(1.0) == b"1.0"
    assert dumps(1.0e-3) == b"0.001"
    assert dumps(True) == b"yes"
    assert dumps(False) == b"no"
    assert dumps("word") == b"word"
    assert dumps(("word", "word")) == b"word word"
    assert dumps('"a string"') == b'"a string"'
    assert dumps(1, kind=Kind.ASCII_FIELD) == b"uniform 1.0"
    assert dumps(1.0, kind=Kind.ASCII_FIELD) == b"uniform 1.0"
    assert dumps(1.0e-3, kind=Kind.ASCII_FIELD) == b"uniform 0.001"
    assert dumps([1.0, 2.0, 3.0]) == b"(1.0 2.0 3.0)"
    assert dumps([1, 2, 3], kind=Kind.ASCII_FIELD) == b"uniform (1.0 2.0 3.0)"
    assert (
        dumps([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kind=Kind.ASCII_FIELD)
        == b"nonuniform List<scalar> 10(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)"
    )
    assert (
        dumps([[1, 2, 3], [4, 5, 6]], kind=Kind.ASCII_FIELD)
        == b"nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
    assert dumps(1, kind=Kind.BINARY_FIELD) == b"uniform 1.0"
    assert dumps(1.0, kind=Kind.BINARY_FIELD) == b"uniform 1.0"
    assert dumps([1, 2, 3], kind=Kind.BINARY_FIELD) == b"uniform (1.0 2.0 3.0)"
    assert (
        dumps([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], kind=Kind.BINARY_FIELD)
        == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
    )
    assert (
        dumps([[1, 2, 3], [4, 5, 6]], kind=Kind.BINARY_FIELD)
        == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )
    assert (
        dumps(np.array([1, 2], dtype=np.float32), kind=Kind.BINARY_FIELD)  # type: ignore [arg-type]
        == b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)"
    )
    assert (
        dumps(FoamFile.DimensionSet(mass=1, length=1, time=-2)) == b"[1 1 -2 0 0 0 0]"
    )
    assert (
        dumps(
            FoamFile.Dimensioned(
                name="g",
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            )
        )
        == b"g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(
            FoamFile.Dimensioned(
                dimensions=FoamFile.DimensionSet(mass=1, length=1, time=-2), value=9.81
            )
        )
        == b"[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]))
        == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert dumps([("a", "b"), ("c", "d")]) == b"(a b; c d;)"
    assert dumps([("a", {"b": "c"}), ("d", {"e": "g"})]) == b"(a {b c;} d {e g;})"
    assert dumps([("a", [0, 1, 2]), ("b", {})]) == b"(a (0 1 2); b {})"
    assert dumps([{"a": "b", "c": "d"}, {"e": "g"}]) == b"({a b; c d;} {e g;})"
    assert dumps(["water", "oil", "mercury", "air"]) == b"(water oil mercury air)"
    assert dumps("div(phi,U)") == b"div(phi,U)"
