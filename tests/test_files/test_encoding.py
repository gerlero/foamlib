import io

import numpy as np
import pytest
from multicollections import MultiDict

from foamlib import Dimensioned, DimensionSet
from foamlib._files._encoding import encoded
from foamlib.typing import Data, FileDict, StandaloneData, SubDict, Tensor


def test_serialize_data() -> None:
    with io.BytesIO() as f:
        encoded(1, f, target=StandaloneData)
        assert f.getvalue() == b"1"
    with io.BytesIO() as f:
        encoded(1.0, f, target=StandaloneData)
        assert f.getvalue() == b"1.0"
    with io.BytesIO() as f:
        encoded(1.0e-3, f, target=StandaloneData)
        assert f.getvalue() == b"0.001"
    with io.BytesIO() as f:
        encoded(True, f, target=StandaloneData)
        assert f.getvalue() == b"yes"
    with io.BytesIO() as f:
        encoded(False, f, target=StandaloneData)
        assert f.getvalue() == b"no"
    with io.BytesIO() as f:
        encoded("word", f, target=StandaloneData)
        assert f.getvalue() == b"word"
    with io.BytesIO() as f:
        encoded(("word", "word"), f, target=StandaloneData)
        assert f.getvalue() == b"word word"
    with io.BytesIO() as f:
        encoded('"a string"', f, target=StandaloneData)
        assert f.getvalue() == b'"a string"'
    with io.BytesIO() as f:
        encoded(1, f, target=Data, keywords=("internalField",))
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        encoded(1.0, f, target=Data, keywords=("internalField",))
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        encoded(1.0e-3, f, target=Data, keywords=("internalField",))
        assert f.getvalue() == b"uniform 0.001"
    with io.BytesIO() as f:
        encoded([1.0, 2.0, 3.0], f, target=StandaloneData)
        assert f.getvalue() == b"(1.0 2.0 3.0)"
    with io.BytesIO() as f:
        encoded([1, 2, 3], f, target=Data, keywords=("internalField",))
        assert f.getvalue() == b"uniform (1.0 2.0 3.0)"
    with io.BytesIO() as f:
        encoded(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f, target=Data, keywords=("internalField",)
        )
        assert (
            f.getvalue()
            == b"nonuniform List<scalar> 10(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)"
        )
    with io.BytesIO() as f:
        encoded([[1, 2, 3], [4, 5, 6]], f, target=Data, keywords=("internalField",))
        assert f.getvalue() == b"nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0))"
    with io.BytesIO() as f:
        encoded(1, f, target=Data, keywords=("internalField",), format_="binary")
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        encoded(1.0, f, target=Data, keywords=("internalField",), format_="binary")
        assert f.getvalue() == b"uniform 1.0"
    with io.BytesIO() as f:
        encoded(
            [1, 2, 3], f, target=Data, keywords=("internalField",), format_="binary"
        )
        assert f.getvalue() == b"uniform (1.0 2.0 3.0)"
    with io.BytesIO() as f:
        encoded(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            f,
            target=Data,
            keywords=("internalField",),
            format_="binary",
        )
        assert (
            f.getvalue()
            == b'nonuniform List<scalar> 10(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@)'
        )
    with io.BytesIO() as f:
        encoded(
            [[1, 2, 3], [4, 5, 6]],
            f,
            target=Data,
            keywords=("internalField",),
            format_="binary",
        )
        assert (
            f.getvalue()
            == b"nonuniform List<vector> 2(\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@)"
        )
    with io.BytesIO() as f:
        encoded(
            np.array([1, 2], dtype=np.float32),
            f,
            target=Data,
            keywords=("internalField",),
            format_="binary",
        )
        assert f.getvalue() == b"nonuniform List<scalar> 2(\x00\x00\x80?\x00\x00\x00@)"
    with io.BytesIO() as f:
        encoded(DimensionSet(mass=1, length=1, time=-2), f, target=StandaloneData)
        assert f.getvalue() == b"[1 1 -2 0 0 0 0]"
    with io.BytesIO() as f:
        encoded(
            Dimensioned(
                name="g",
                dimensions=DimensionSet(mass=1, length=1, time=-2),
                value=9.81,
            ),
            f,
            target=StandaloneData,
        )
        assert f.getvalue() == b"g [1 1 -2 0 0 0 0] 9.81"
    with io.BytesIO() as f:
        encoded(
            Dimensioned(dimensions=DimensionSet(mass=1, length=1, time=-2), value=9.81),
            f,
            target=StandaloneData,
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
            target=StandaloneData,
        )
        assert f.getvalue() == b"hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)"
    with io.BytesIO() as f:
        encoded(
            [("a", "b"), ("c", "d"), ("n", False), ("y", True)],
            f,
            target=StandaloneData,
        )
        assert f.getvalue() == b"(a b; c d; n no; y yes;)"
    with io.BytesIO() as f:
        encoded([("a", {"b": "c"}), ("d", {"e": "g"})], f, target=StandaloneData)
        assert f.getvalue() == b"(a {b c;} d {e g;})"
    with io.BytesIO() as f:
        encoded([("a", [0, 1, 2]), ("b", {})], f, target=StandaloneData)
        assert f.getvalue() == b"(a (0 1 2); b {})"
    with io.BytesIO() as f:
        encoded(["water", "oil", "mercury", "air"], f, target=StandaloneData)
        assert f.getvalue() == b"(water oil mercury air)"
    with io.BytesIO() as f:
        encoded("div(phi,U)", f, target=StandaloneData)
        assert f.getvalue() == b"div(phi,U)"


def test_faces_like_list() -> None:
    faces_like_list = encoded([[1, 2, 3], [4, 5, 6, 7]])
    assert isinstance(faces_like_list, list)
    assert isinstance(faces_like_list[0], np.ndarray)
    assert faces_like_list[0].dtype == int
    assert faces_like_list[0].tolist() == [1, 2, 3]
    assert isinstance(faces_like_list[1], np.ndarray)
    assert faces_like_list[1].dtype == int
    assert faces_like_list[1].tolist() == [4, 5, 6, 7]
    with io.BytesIO() as f:
        encoded(faces_like_list, f, target=StandaloneData)
        assert f.getvalue() == b"(3(1 2 3) 4(4 5 6 7))"


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


def test_token_without_writer() -> None:
    assert encoded("myKeyword", target=str) == "myKeyword"


def test_token_with_writer() -> None:
    with io.BytesIO() as fp:
        assert encoded("myKeyword", fp, target=str) == "myKeyword"
        assert fp.getvalue() == b"myKeyword"


def test_tensor_without_writer() -> None:
    np.testing.assert_array_equal(
        encoded([1, 2, 3], target=Tensor), np.array([1.0, 2.0, 3.0])
    )


def test_tensor_with_writer() -> None:
    with io.BytesIO() as fp:
        result = encoded([1, 2, 3], fp, target=Tensor)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
        assert fp.getvalue() == b"(1.0 2.0 3.0)"


def test_data_with_writer() -> None:
    with io.BytesIO() as fp:
        assert encoded(("uniform", 3), fp, target=Data, keywords=("inlet",)) == (
            "uniform",
            3,
        )
        assert fp.getvalue() == b"uniform 3"


def test_file_dict_ascii() -> None:
    with io.BytesIO() as fp:
        result = encoded(
            {"FoamFile": {"format": "ascii"}, "internalField": [1, 2, 3, 4]},
            fp,
            target=FileDict,
        )
        np.testing.assert_array_equal(result["internalField"], [1.0, 2.0, 3.0, 4.0])
        assert fp.getvalue() == (
            b"FoamFile {format ascii;} "
            b"internalField nonuniform List<scalar> 4(1.0 2.0 3.0 4.0);"
        )


def test_file_dict_binary_preserves_float32() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with io.BytesIO() as fp:
        result = encoded(
            {"FoamFile": {"format": "binary"}, "internalField": values},
            fp,
            target=FileDict,
        )
        assert result["internalField"] is values
        assert fp.getvalue() == (
            b"FoamFile {format binary;} "
            b"internalField nonuniform List<scalar> 4(" + values.tobytes() + b");"
        )


def test_file_dict_ascii_loses_float32() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with io.BytesIO() as fp:
        result = encoded(
            {"internalField": values},
            fp,
            target=FileDict,
        )
        internal_field = result["internalField"]
        assert internal_field is not values
        assert isinstance(internal_field, np.ndarray)
        assert internal_field.dtype == np.float64
        assert (internal_field == values).all()
        assert fp.getvalue() == (
            b"internalField nonuniform List<scalar> 4(1.0 2.0 3.0 4.0);"
        )


def test_standalone_integer_array_binary() -> None:
    values = np.array([1, 2, 3], dtype=np.int32)
    with io.BytesIO() as fp:
        result = encoded(values, fp, target=StandaloneData, format_="binary")
        assert result is values
        assert fp.getvalue() == b"3(" + values.tobytes() + b")"


def test_nested_subdict() -> None:
    with io.BytesIO() as fp:
        result = encoded(
            {"inlet": {"type": "fixedValue", "value": 3}},
            fp,
            target=SubDict,
            keywords=("boundaryField",),
        )
        assert isinstance(result, dict)
        assert result == {"inlet": {"type": "fixedValue", "value": 3}}
        assert fp.getvalue() == b"{inlet {type fixedValue; value uniform 3.0;}}"


def test_list_of_keyword_entries() -> None:
    with io.BytesIO() as fp:
        result = encoded([("a", 1), ("b", 2)], fp)
        assert result == [("a", 1), ("b", 2)]
        assert fp.getvalue() == b"(a 1; b 2;)"


def test_duplicate_keywords_are_not_written_twice() -> None:
    data: FileDict = MultiDict([("a", 1), ("b", 2), ("a", 3)])
    with io.BytesIO() as fp:
        with pytest.warns(UserWarning, match="Duplicate file keyword"):
            result = encoded(data, fp, target=FileDict)
        assert list(result.items()) == [("b", 2), ("a", 3)]
        assert fp.getvalue() == b"b 2; a 3;"


def test_directive_with_none_value() -> None:
    with io.BytesIO() as fp:
        result = encoded({"#include": '"otherFile"'}, fp, target=FileDict)
        assert result == {"#include": '"otherFile"'}
        assert fp.getvalue() == b'\n#include "otherFile"\n'


def test_invalid_keyword_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        encoded({"bad key": 1}, target=FileDict)
    with pytest.raises(ValueError, match="invalid"):
        encoded({" badkey": 1}, target=FileDict)
    with pytest.raises(ValueError, match="invalid"):
        encoded({"badkey ": 1}, target=FileDict)


def test_reject_mapping_as_directive_value() -> None:
    with pytest.raises(TypeError, match="cannot have a mapping"):
        encoded({"#include": {}}, target=FileDict)
