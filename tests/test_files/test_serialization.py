import numpy as np
import pytest

from foamlib import Dimensioned, DimensionSet
from foamlib._files._normalization import normalized
from foamlib._files._serialization import dumps
from foamlib.typing import Data, FileDict


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
            normalized(1, target=Data, keywords=("internalField",)),
            keywords=("internalField",),
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0, target=Data, keywords=("internalField",)),
            keywords=("internalField",),
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0e-3, target=Data, keywords=("internalField",)),
            keywords=("internalField",),
        )
        == b"uniform 0.001"
    )
    assert dumps(normalized([1.0, 2.0, 3.0])) == b"(1.0 2.0 3.0)"
    assert (
        dumps(
            normalized([1, 2, 3], target=Data, keywords=("internalField",)),
            keywords=("internalField",),
        )
        == b"uniform (1.0 2.0 3.0)"
    )
    assert (
        dumps(
            normalized(
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
            normalized(
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
            normalized(1, target=Data, keywords=("internalField",), binary=True),
            keywords=("internalField",),
            format_="binary",
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(1.0, target=Data, keywords=("internalField",), binary=True),
            keywords=("internalField",),
            format_="binary",
        )
        == b"uniform 1.0"
    )
    assert (
        dumps(
            normalized(
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
            normalized(
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
            normalized(
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
            normalized(
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
    assert faces_like_list[0].tolist() == [1, 2, 3]
    assert isinstance(faces_like_list[1], np.ndarray)
    assert faces_like_list[1].dtype == int
    assert faces_like_list[1].tolist() == [4, 5, 6, 7]
    assert dumps(faces_like_list) == b"(3(1 2 3) 4(4 5 6 7))"


def test_bool_tokens() -> None:
    with pytest.warns(UserWarning):
        assert normalized("no") is False
    with pytest.warns(UserWarning):
        assert normalized("yes") is True

    assert normalized("no", target=str) == "no"
    assert normalized("yes", target=str) == "yes"

    with pytest.warns(UserWarning):
        assert normalized({"no": "no", "yes": "yes"}, target=FileDict) == {
            "no": False,
            "yes": True,
        }

    with pytest.raises(TypeError, match="False"):
        normalized({False: True}, target=FileDict)  # ty: ignore[no-matching-overload]
