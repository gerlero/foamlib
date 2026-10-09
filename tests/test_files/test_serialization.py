import io

import numpy as np
import pytest

from foamlib import Dimensioned, DimensionSet
from foamlib._files._encoding import encoded
from foamlib.typing import Data, FileDict


def test_serialize_data() -> None:
    with io.BytesIO() as f:
        assert encoded(1, f) == 1
        assert f.getvalue() == b"1"
    with io.BytesIO() as f:
        assert encoded(1.0, f) == 1.0
        assert f.getvalue() == b"1.0"
    with io.BytesIO() as f:
        assert encoded(1.0e-3, f) == 0.001
        assert f.getvalue() == b"0.001"
    with io.BytesIO() as f:
        assert encoded(True, f) is True
        assert f.getvalue() == b"yes"
    with io.BytesIO() as f:
        assert encoded(False, f) is False
        assert f.getvalue() == b"no"
    with io.BytesIO() as f:
        assert encoded("word", f) == "word"
        assert f.getvalue() == b"word"
    with io.BytesIO() as f:
        assert encoded(("word", "word"), f) == ("word", "word")
        assert f.getvalue() == b"word word"
    with io.BytesIO() as f:
        assert encoded('"a string"', f) == '"a string"'
        assert f.getvalue() == b'"a string"'
    with io.BytesIO() as f:
        assert (
            encoded(1, f, target=Data, keywords=("internalField",)) == 1.0  # ty: ignore[no-matching-overload]
        )
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        assert encoded(1.0, f, target=Data, keywords=("internalField",)) == 1.0
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        assert encoded(1.0e-3, f, target=Data, keywords=("internalField",)) == 0.001
        assert f.getvalue() == b"uniform 0.001"
    with io.BytesIO() as f:
        assert np.array_equal(encoded([1.0, 2.0, 3.0], f), np.array([1.0, 2.0, 3.0]))
        assert f.getvalue() == b"(1.0 2.0 3.0)"
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded([1, 2, 3], f, target=Data, keywords=("internalField",)),  # ty: ignore[no-matching-overload]
            np.array([1.0, 2.0, 3.0]),
        )
        assert f.getvalue() == b"uniform (1.0 2.0 3.0)"
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                f,
                target=Data,
                keywords=("internalField",),
            ),  # ty: ignore[no-matching-overload]
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
        assert (
            f.getvalue()
            == b"nonuniform List<scalar> 10(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)"
        )
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded(
                [[1, 2, 3], [4, 5, 6]],
                f,
                target=Data,
                keywords=("internalField",),
            ),  # ty: ignore[no-matching-overload]
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        assert f.getvalue() == b"nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0))"
    with io.BytesIO() as f:
        assert (
            encoded(1, f, target=Data, keywords=("internalField",), format_="binary")  # ty: ignore[no-matching-overload]
            == 1.0
        )
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        assert (
            encoded(1.0, f, target=Data, keywords=("internalField",), format_="binary")
            == 1.0
        )
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded(
                [1, 2, 3],
                f,
                target=Data,
                keywords=("internalField",),
                format_="binary",
            ),  # ty: ignore[no-matching-overload]
            np.array([1.0, 2.0, 3.0]),
        )
        assert f.getvalue() == b"uniform (1.0 2.0 3.0)"
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                f,
                target=Data,
                keywords=("internalField",),
                format_="binary",
            ),  # ty: ignore[no-matching-overload]
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
        assert (
            f.getvalue()
            == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
        )
    with io.BytesIO() as f:
        assert np.array_equal(
            encoded(
                [[1, 2, 3], [4, 5, 6]],
                f,
                target=Data,
                keywords=("internalField",),
                format_="binary",
            ),  # ty: ignore[no-matching-overload]
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        assert (
            f.getvalue()
            == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
        )
    with io.BytesIO() as f:
        assert (
            encoded(
                np.array([1, 2], dtype=np.float32),
                f,
                target=Data,
                keywords=("internalField",),
                format_="binary",
            )  # ty: ignore[no-matching-overload]
            is not None
        )
        assert f.getvalue() == b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)"
    with io.BytesIO() as f:
        assert encoded(DimensionSet(mass=1, length=1, time=-2), f) == DimensionSet(
            mass=1, length=1, time=-2
        )
        assert f.getvalue() == b"[1 1 -2 0 0 0 0]"
    with io.BytesIO() as f:
        encoded(
            Dimensioned(
                name="g",
                dimensions=DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            ),
            f,
        )
        assert f.getvalue() == b"g [1 1 -2 0 0 0 0] 9.81"
    with io.BytesIO() as f:
        encoded(
            Dimensioned(dimensions=DimensionSet(mass=1, length=1, time=-2), value=9.81),
            f,
        )
        assert f.getvalue() == b"[1 1 -2 0 0 0 0] 9.81"
    with io.BytesIO() as f:
        encoded(
            (
                "hex",
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 1, 1],
                "simpleGrading",
                [1, 1, 1],
            ),
            f,
        )
        assert f.getvalue() == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    with io.BytesIO() as f:
        encoded([("a", "b"), ("c", "d"), ("n", False), ("y", True)], f)
        assert f.getvalue() == b"(a b; c d; n no; y yes;)"
    with io.BytesIO() as f:
        encoded([("a", {"b": "c"}), ("d", {"e": "g"})], f)
        assert f.getvalue() == b"(a {b c;} d {e g;})"
    with io.BytesIO() as f:
        encoded([("a", [0, 1, 2]), ("b", {})], f)
        assert f.getvalue() == b"(a (0 1 2); b {})"
    with io.BytesIO() as f:
        encoded(["water", "oil", "mercury", "air"], f)
        assert f.getvalue() == b"(water oil mercury air)"
    with io.BytesIO() as f:
        assert encoded("div(phi,U)", f) == "div(phi,U)"
        assert f.getvalue() == b"div(phi,U)"


def test_faces_like_list() -> None:
    with io.BytesIO() as f:
        faces_like_list = encoded([[1, 2, 3], [4, 5, 6, 7]], f)
        assert f.getvalue() == b"(3(1 2 3) 4(4 5 6 7))"
    assert isinstance(faces_like_list, list)
    assert isinstance(faces_like_list[0], np.ndarray)
    assert faces_like_list[0].dtype == int
    assert faces_like_list[0].tolist() == [1, 2, 3]
    assert isinstance(faces_like_list[1], np.ndarray)
    assert faces_like_list[1].dtype == int
    assert faces_like_list[1].tolist() == [4, 5, 6, 7]


def test_bool_tokens() -> None:
    with pytest.warns(UserWarning):
        assert encoded("no") is False
    with pytest.warns(UserWarning):
        assert encoded("yes") is True

    assert encoded("no", target=str) == "no"
    assert encoded("yes", target=str) == "yes"

    with pytest.warns(UserWarning):
        assert encoded({"no": "no", "yes": "yes"}, target=FileDict) == {
            "no": False,
            "yes": True,
        }

    with pytest.raises(TypeError, match="False"):
        encoded({False: True}, target=FileDict)  # ty: ignore[no-matching-overload]
