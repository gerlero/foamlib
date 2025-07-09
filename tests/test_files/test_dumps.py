import numpy as np
from foamlib import FoamFile
from foamlib._files._serialization import dumps


def test_serialize_data() -> None:
    assert dumps(1) == b"1"
    assert dumps(1.0) == b"1.0"
    assert dumps(1.0e-3) == b"0.001"
    assert dumps(True) == b"yes"
    assert dumps(False) == b"no"
    assert dumps("word") == b"word"
    assert dumps(("word", "word"), keywords=()) == b"word word"
    assert dumps('"a string"') == b'"a string"'
    assert dumps(1, keywords=("internalField",)) == b"uniform 1.0"
    assert dumps(1.0, keywords=("internalField",)) == b"uniform 1.0"
    assert dumps(1.0e-3, keywords=("internalField",)) == b"uniform 0.001"
    assert dumps([1.0, 2.0, 3.0]) == b"(1.0 2.0 3.0)"
    assert dumps([1, 2, 3], keywords=("internalField",)) == b"uniform (1.0 2.0 3.0)"
    assert (
        dumps([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], keywords=("internalField",))
        == b"nonuniform List<scalar> 10(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)"
    )
    assert (
        dumps([[1, 2, 3], [4, 5, 6]], keywords=("internalField",))
        == b"nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
    assert (
        dumps(1, keywords=("internalField",), header={"format": "binary"})
        == b"uniform 1.0"
    )
    assert (
        dumps(1.0, keywords=("internalField",), header={"format": "binary"})
        == b"uniform 1.0"
    )
    assert (
        dumps([1, 2, 3], keywords=("internalField",), header={"format": "binary"})
        == b"uniform (1.0 2.0 3.0)"
    )
    assert (
        dumps(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            keywords=("internalField",),
            header={"format": "binary"},
        )
        == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
    )
    assert (
        dumps(
            [[1, 2, 3], [4, 5, 6]],
            keywords=("internalField",),
            header={"format": "binary"},
        )
        == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )
    assert (
        dumps(
            np.array([1, 2], dtype=np.float32),
            keywords=("internalField",),
            header={"format": "binary"},
        )
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
        dumps(
            ("hex", [0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1], "simpleGrading", [1, 1, 1]),
            keywords=(),
        )
        == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert (
        dumps([("a", "b"), ("c", "d"), ("n", "no"), ("y", "yes")])
        == b"(a b; c d; n no; y yes;)"
    )
    assert dumps([("a", {"b": "c"}), ("d", {"e": "g"})]) == b"(a {b c;} d {e g;})"
    assert dumps([("a", [0, 1, 2]), ("b", {})]) == b"(a (0 1 2); b {})"
    assert dumps(["water", "oil", "mercury", "air"]) == b"(water oil mercury air)"
    assert dumps("div(phi,U)") == b"div(phi,U)"


def test_serialize_file() -> None:
    assert FoamFile.dumps(1.0, ensure_header=False) == b"1.0"
    assert (
        FoamFile.dumps(1.0)
        == b"{FoamFile {version 2.0; format ascii; class dictionary;}} 1.0"
    )
    assert (
        FoamFile.dumps({"a": "b", "c": "d", "n": "no", "y": "yes"})
        == b"{FoamFile {version 2.0; format ascii; class dictionary;}} a b; c d; n no; y yes;"
    )
    assert (
        FoamFile.dumps({"internalField": [[1, 2, 3], [4, 5, 6]]})
        == b"{FoamFile {version 2.0; format ascii; class volVectorField;}} internalField nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0));"
    )
    assert (
        FoamFile.dumps([[1, 2, 3], [4, 5, 6]])
        == b"{FoamFile {version 2.0; format ascii; class dictionary;}} ((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
    assert FoamFile.dumps([1, 2, 3], ensure_header=False) == b"(1 2 3)"
    assert (
        FoamFile.dumps(
            {
                "FoamFile": {"format": "binary"},
                None: np.array([1, 2, 3], dtype=np.int32),
            },
            ensure_header=False,
        )
        == b"FoamFile {format binary;} 3(\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00)"
    )
    assert (
        FoamFile.dumps({"#include": "$FOAM_CASE/simControls"}, ensure_header=False)
        == b"\n#include $FOAM_CASE/simControls\n"
    )
    indices = np.array([0, 3, 7], dtype=np.int32)
    values = np.array(
        [904040, 904479, 924424, 3516631, 3516634, 3516633, 3516632], dtype=np.int32
    )
    assert (
        FoamFile.dumps(
            {"FoamFile": {"format": "binary"}, None: (indices, values)},
            ensure_header=False,
        )
        == b"FoamFile {format binary;} 3(\x00\x00\x00\x00\x03\x00\x00\x00\x07\x00\x00\x00) 7(h\xcb\r\x00\x1f\xcd\r\x00\x08\x1b\x0e\x00\xd7\xa85\x00\xda\xa85\x00\xd9\xa85\x00\xd8\xa85\x00)"
    )
