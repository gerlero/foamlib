import numpy as np
from foamlib import Dimensioned, DimensionSet, FoamFile
from foamlib._files._normalization import normalized
from foamlib._files._serialization import dumps
from foamlib.typing import Data
from multicollections import MultiDict


def test_serialize_data() -> None:
    assert dumps(normalized(1)) == b"1"
    assert dumps(normalized(1.0)) == b"1.0"
    assert dumps(normalized(1.0e-3)) == b"0.001"
    assert dumps(normalized(True)) == b"yes"
    assert dumps(normalized(False)) == b"no"
    assert dumps(normalized("word")) == b"word"
    assert dumps(normalized(("word", "word"))) == b"word word"
    assert dumps(normalized('"a string"')) == b'"a string"'
    assert (
        dumps(
            normalized(1, target=Data, keywords=("internalField",)),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0, target=Data, keywords=("internalField",)),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0e-3, target=Data, keywords=("internalField",)),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
        )
        == b"uniform 0.001"
    )
    assert dumps(normalized([1.0, 2.0, 3.0])) == b"(1.0 2.0 3.0)"
    assert (
        dumps(
            normalized([1, 2, 3], target=Data, keywords=("internalField",)),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
        )
        == b"uniform (1.0 2.0 3.0)"
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                target=Data,
                keywords=("internalField",),
            ),
            keywords=("internalField",),
        )
        == b"nonuniform List<scalar> 10(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)"
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                [[1, 2, 3], [4, 5, 6]],
                target=Data,
                keywords=("internalField",),
            ),
            keywords=("internalField",),
        )
        == b"nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
    assert (
        dumps(
            normalized(1, target=Data, keywords=("internalField",), binary=True),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
            format_="binary",
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0, target=Data, keywords=("internalField",), binary=True),  # ty: ignore[no-matching-overload]
            keywords=("internalField",),
            format_="binary",
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                [1, 2, 3],
                target=Data,
                keywords=("internalField",),
                binary=True,
            ),
            keywords=("internalField",),
            format_="binary",
        )
        == b"uniform (1.0 2.0 3.0)"
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                target=Data,
                keywords=("internalField",),
                binary=True,
            ),
            keywords=("internalField",),
            format_="binary",
        )
        == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                [[1, 2, 3], [4, 5, 6]],
                target=Data,
                keywords=("internalField",),
                binary=True,
            ),
            keywords=("internalField",),
            format_="binary",
        )
        == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
    )
    assert (
        dumps(
            normalized(  # ty: ignore[no-matching-overload]
                np.array([1, 2], dtype=np.float32),
                target=Data,
                keywords=("internalField",),
                binary=True,
            ),
            keywords=("internalField",),
            format_="binary",
        )
        == b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)"
    )
    assert (
        dumps(normalized(DimensionSet(mass=1, length=1, time=-2)))
        == b"[1 1 -2 0 0 0 0]"
    )
    assert (
        dumps(
            normalized(
                Dimensioned(
                    name="g",
                    dimensions=DimensionSet(mass=1, length=1, time=-2),
                    value=9.81,
                )
            )
        )
        == b"g [1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(
            normalized(
                Dimensioned(
                    dimensions=DimensionSet(mass=1, length=1, time=-2), value=9.81
                )
            )
        )
        == b"[1 1 -2 0 0 0 0] 9.81"
    )
    assert (
        dumps(
            normalized(
                (
                    "hex",
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [1, 1, 1],
                    "simpleGrading",
                    [1, 1, 1],
                ),
            )
        )
        == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    )
    assert (
        dumps(normalized([("a", "b"), ("c", "d"), ("n", False), ("y", True)]))
        == b"(a b; c d; n no; y yes;)"
    )
    assert (
        dumps(normalized([("a", {"b": "c"}), ("d", {"e": "g"})]))
        == b"(a {b c;} d {e g;})"
    )
    assert dumps(normalized([("a", [0, 1, 2]), ("b", {})])) == b"(a (0 1 2); b {})"
    assert (
        dumps(normalized(["water", "oil", "mercury", "air"]))
        == b"(water oil mercury air)"
    )
    assert dumps(normalized("div(phi,U)")) == b"div(phi,U)"


def test_faces_like_list() -> None:
    faces_like_list = normalized([[1, 2, 3], [4, 5, 6, 7]])
    assert isinstance(faces_like_list, list)
    assert isinstance(faces_like_list[0], np.ndarray)
    assert faces_like_list[0].dtype == int
    assert faces_like_list[0].tolist() == [1, 2, 3]  # ty: ignore[no-matching-overload]
    assert isinstance(faces_like_list[1], np.ndarray)
    assert faces_like_list[1].dtype == int
    assert faces_like_list[1].tolist() == [4, 5, 6, 7]  # ty: ignore[no-matching-overload]
    assert dumps(faces_like_list) == b"(3(1 2 3) 4(4 5 6 7))"


def test_serialize_file() -> None:
    assert FoamFile.dumps(1.0, ensure_header=False) == b"1.0"
    assert (
        FoamFile.dumps(1.0)
        == b"FoamFile {version 2.0; format ascii; class dictionary;} 1.0"
    )
    assert (
        FoamFile.dumps({"a": "b", "c": "d", "n": False, "y": True})
        == b"FoamFile {version 2.0; format ascii; class dictionary;} a b; c d; n no; y yes;"
    )
    assert (
        FoamFile.dumps({"internalField": [[1, 2, 3], [4, 5, 6]]})
        == b"FoamFile {version 2.0; format ascii; class volVectorField;} internalField nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0));"
    )
    assert (
        FoamFile.dumps([1, 2, 3, 4, 5, 6])
        == b"FoamFile {version 2.0; format ascii; class dictionary;} (1 2 3 4 5 6)"
    )
    assert FoamFile.dumps([1, 2, 3], ensure_header=False) == b"(1 2 3)"
    assert FoamFile.dumps([1.0, 2, 3], ensure_header=False) == b"(1.0 2.0 3.0)"
    assert (
        FoamFile.dumps([[1, 2, 3], [4, 5.0, 6]], ensure_header=False)
        == b"((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
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
    assert (
        FoamFile.dumps([[1, 2, 3, 4], [5, 6, 7, 8]], ensure_header=False)
        == b"(4(1 2 3 4) 4(5 6 7 8))"
    )
    assert (
        FoamFile.dumps([[1, 2, 3], [4, 5, 6, 7]], ensure_header=False)
        == b"(3(1 2 3) 4(4 5 6 7))"
    )
    assert (
        FoamFile.dumps([[1, 2, 3], [4, 5, 6]], ensure_header=False)
        == b"(3(1 2 3) 3(4 5 6))"
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
    assert (
        FoamFile.dumps(
            MultiDict([("#includeFunc", '"func1"'), ("#includeFunc", '"func2"')]),
            ensure_header=False,
        )
        == b'\n#includeFunc "func1"\n \n#includeFunc "func2"\n'
    )
    assert FoamFile.dumps({"keyword": None}, ensure_header=False) == b"keyword;"
