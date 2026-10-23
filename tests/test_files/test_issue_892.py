from pathlib import Path

import numpy as np
import pytest

from foamlib import FoamFile, FoamFileDecodeError, dumps, loads
from foamlib._files._encoding import encoded
from foamlib.typing import FileDict, FileDictLike


def _zone_contents(
    kind: str,
    format_: str,
    label_dtype: type[np.int32] | type[np.int64],
    *,
    meta: bool,
) -> bytes:
    header = (
        f"FoamFile {{version 2.0; format {format_}; "
        f"class regIOobject; object {kind}Zones;}}\n"
    ).encode()
    labels = np.array([0, 41, 125], dtype=label_dtype)
    flips = np.array([False, True, False], dtype=np.bool_)
    if format_ == "binary":
        label_data = b"3(" + labels.tobytes() + b")"
        flip_data = b"3(" + flips.tobytes() + b")"
        empty = b"0"
    else:
        label_data = b"3(0 41 125)"
        flip_data = b"3(0 1 0)"
        empty = b"0()"
    label_key = f"{kind}Labels".encode()
    zone_type = f"type {kind}Zone; ".encode()
    body = (
        b"2(\nfirst {" + zone_type + label_key + b" List<label> " + label_data + b"; "
    )
    if kind == "face":
        body += b"flipMap List<bool> " + flip_data + b"; "
    body += b"}\nempty {" + zone_type + label_key + b" List<label> " + empty + b"; }\n)"
    if meta:
        header += b"meta {names 2(first empty);}\n"
    return header + body


def _assert_zones(value: object, kind: str) -> None:
    assert isinstance(value, list)
    assert len(value) == 2
    first_name, first = value[0]
    empty_name, empty = value[1]
    assert first_name == "first"
    assert empty_name == "empty"
    assert first["type"] == empty["type"] == f"{kind}Zone"
    labels = first[f"{kind}Labels"]
    assert isinstance(labels, np.ndarray)
    assert labels.dtype.kind == "i"
    assert labels.shape == (3,)
    assert np.all(labels == [0, 41, 125])
    empty_labels = empty[f"{kind}Labels"]
    assert isinstance(empty_labels, np.ndarray)
    assert empty_labels.dtype.kind == "i"
    assert empty_labels.shape == (0,)
    if kind == "face":
        flips = first["flipMap"]
        assert isinstance(flips, np.ndarray)
        assert flips.dtype == np.bool_
        assert flips.itemsize == 1
        assert flips.shape == (3,)
        assert np.all(flips == [False, True, False])
        assert "flipMap" not in empty


def _first_zone(value: object) -> dict:
    assert isinstance(value, list)
    first = value[0]
    assert isinstance(first, tuple)
    zone = first[1]
    assert isinstance(zone, dict)
    return zone


@pytest.mark.parametrize("kind", ["cell", "face", "point"])
@pytest.mark.parametrize("format_", ["ascii", "binary"])
@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("meta", [False, True])
def test_zone_lists(
    tmp_path: Path,
    kind: str,
    format_: str,
    label_dtype: type[np.int32] | type[np.int64],
    *,
    meta: bool,
) -> None:
    contents = _zone_contents(kind, format_, label_dtype, meta=meta)
    parsed = loads(contents, include_header=True)
    assert isinstance(parsed, dict)
    _assert_zones(parsed[None], kind)
    if format_ == "binary":
        assert _first_zone(parsed[None])[f"{kind}Labels"].dtype == label_dtype
    if meta:
        assert parsed["meta"] == {"names": ["first", "empty"]}
    else:
        assert "meta" not in parsed

    path = tmp_path / f"{kind}Zones"
    path.write_bytes(contents)
    file = FoamFile(path)
    _assert_zones(file[None], kind)
    assert file.class_ == "regIOobject"
    assert file.object_ == f"{kind}Zones"

    serialized = dumps(parsed)
    assert b"List<label>" in serialized
    if kind == "face":
        assert b"List<bool>" in serialized
        if format_ == "binary":
            assert b"3(\x00\x01\x00)" in serialized
    restored = loads(serialized, include_header=True)
    assert isinstance(restored, dict)
    _assert_zones(restored[None], kind)
    if format_ == "binary":
        assert _first_zone(restored[None])[f"{kind}Labels"].dtype == label_dtype
    if meta:
        assert restored["meta"] == parsed["meta"]


@pytest.mark.parametrize("kind", ["cell", "face", "point"])
def test_zone_lists_without_type_tokens(kind: str) -> None:
    contents = _zone_contents(kind, "ascii", np.int64, meta=False)
    contents = contents.replace(b"List<label> ", b"").replace(b"List<bool> ", b"")
    _assert_zones(loads(contents), kind)


@pytest.mark.parametrize("kind", ["cell", "face", "point"])
def test_zone_lists_without_type_entries(kind: str) -> None:
    contents = _zone_contents(kind, "binary", np.int32, meta=False)
    contents = contents.replace(f"type {kind}Zone; ".encode(), b"")
    parsed = loads(contents)
    first = _first_zone(parsed)
    assert "type" not in first
    labels = first[f"{kind}Labels"]
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (3,)
    assert np.all(labels == [0, 41, 125])


@pytest.mark.parametrize("format_", ["ascii", "binary"])
@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
def test_zone_list_normalization_roundtrip(
    format_: str, label_dtype: type[np.int32] | type[np.int64]
) -> None:
    value: FileDictLike = {
        "FoamFile": {"format": format_},
        None: [
            (
                "zone",
                {
                    "faceLabels": np.array([0, 41, 125], dtype=label_dtype),
                    "flipMap": [False, True, False],
                },
            )
        ],
    }
    normalized = encoded(value, target=FileDict)
    restored = loads(dumps(normalized), include_header=True)
    assert isinstance(restored, dict)
    for stage in (normalized, restored):
        zone = _first_zone(stage[None])
        labels = zone["faceLabels"]
        assert isinstance(labels, np.ndarray)
        assert labels.dtype.kind == "i"
        if format_ == "binary":
            assert labels.dtype == label_dtype
        assert labels.shape == (3,)
        assert np.all(labels == [0, 41, 125])
        flips = zone["flipMap"]
        assert isinstance(flips, np.ndarray)
        assert flips.dtype == np.bool_
        assert flips.shape == (3,)
        assert np.all(flips == [False, True, False])


@pytest.mark.parametrize(
    ("source", "expected", "dtype"),
    [
        ("List<label> 3(0 41 125)", [0, 41, 125], np.int64),
        ("List<label> (0 41 125)", [0, 41, 125], np.int64),
        ("List<label> 3{7}", [7, 7, 7], np.int64),
        ("List<label> 0()", [], np.int64),
        ("List<label> 0", [], np.int64),
        ("List<label> 0{7}", [], np.int64),
        ("List<bool> 3(0 1 0)", [False, True, False], np.bool_),
        ("List<bool> (0 1 0)", [False, True, False], np.bool_),
        ("List<bool> 3{1}", [True, True, True], np.bool_),
        (
            "List<bool> 8(true false yes no on off 0 1)",
            [True, False, True, False, True, False, False, True],
            np.bool_,
        ),
        ("List<bool> 0()", [], np.bool_),
        ("List<bool> 0", [], np.bool_),
        ("List<bool> 0{0}", [], np.bool_),
    ],
)
def test_standalone_typed_lists(
    source: str,
    expected: list[int | bool],
    dtype: type[np.int64] | type[np.bool_],
) -> None:
    parsed = loads(source)
    assert isinstance(parsed, np.ndarray)
    assert parsed.dtype == dtype
    assert parsed.shape == (len(expected),)
    assert np.all(parsed == expected)


@pytest.mark.parametrize("type_name", ["label", "bool"])
def test_binary_empty_list_without_parentheses(type_name: str) -> None:
    parsed = loads(
        f"FoamFile {{format binary;}} values List<{type_name}> 0; after preserved;"
    )
    assert isinstance(parsed, dict)
    values = parsed["values"]
    assert isinstance(values, np.ndarray)
    assert values.shape == (0,)
    assert values.dtype.kind == ("b" if type_name == "bool" else "i")
    assert parsed["after"] == "preserved"


@pytest.mark.parametrize("type_name", ["label", "bool"])
@pytest.mark.parametrize("format_", ["ascii", "binary"])
def test_standalone_empty_typed_list_roundtrip(type_name: str, format_: str) -> None:
    parsed = loads(
        f"FoamFile {{format {format_};}} List<{type_name}> 0", include_header=True
    )
    serialized = dumps(parsed)
    assert f"List<{type_name}>".encode() in serialized
    restored = loads(serialized)
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == (np.int64 if type_name == "label" else np.bool_)
    assert restored.shape == (0,)


@pytest.mark.parametrize("format_", ["ascii", "binary"])
def test_typed_array_roundtrip(format_: str) -> None:
    labels = np.array([0, 41, 125], dtype=np.int32)
    flips = np.array([False, True, False], dtype=np.bool_)
    serialized = dumps(
        {
            "FoamFile": {"format": format_},
            "labels": labels,
            "flips": flips,
            "emptyLabels": np.array([], dtype=np.int64),
            "emptyFlips": np.array([], dtype=np.bool_),
            "explicitLabels": ("List<label>", [3, 4]),
            "explicitFlips": ("List<bool>", [True, False]),
        }
    )
    assert b"List<label>" in serialized
    assert b"List<bool>" in serialized
    parsed = loads(serialized)
    assert isinstance(parsed, dict)
    for key, expected_labels in (
        ("labels", [0, 41, 125]),
        ("emptyLabels", []),
        ("explicitLabels", [3, 4]),
    ):
        labels_array = parsed[key]
        assert isinstance(labels_array, np.ndarray)
        assert labels_array.dtype == (
            np.int32 if key == "labels" and format_ == "binary" else np.int64
        )
        assert labels_array.shape == (len(expected_labels),)
        assert np.all(labels_array == expected_labels)
    for key, expected_flips in (
        ("flips", [False, True, False]),
        ("emptyFlips", []),
        ("explicitFlips", [True, False]),
    ):
        flips_array = parsed[key]
        assert isinstance(flips_array, np.ndarray)
        assert flips_array.dtype == np.bool_
        assert flips_array.shape == (len(expected_flips),)
        assert np.all(flips_array == expected_flips)


def test_edit_zone_lists(tmp_path: Path) -> None:
    path = tmp_path / "faceZones"
    path.write_bytes(_zone_contents("face", "ascii", np.int64, meta=True))
    file = FoamFile(path)
    zones = file[None]
    _assert_zones(zones, "face")

    file.format = "binary"
    file[None] = zones
    assert b"List<bool> 3(\x00\x01\x00)" in path.read_bytes()
    _assert_zones(FoamFile(path)[None], "face")

    first = _first_zone(zones)
    first["faceLabels"] = np.array([1, 2], dtype=np.int32)
    first["flipMap"] = np.array([True, False], dtype=np.bool_)
    file[None] = zones
    reopened = FoamFile(path)
    updated = _first_zone(reopened[None])
    updated_labels = updated["faceLabels"]
    assert isinstance(updated_labels, np.ndarray)
    assert updated_labels.dtype == np.int32
    assert updated_labels.shape == (2,)
    assert np.all(updated_labels == [1, 2])
    updated_flips = updated["flipMap"]
    assert isinstance(updated_flips, np.ndarray)
    assert updated_flips.dtype == np.bool_
    assert updated_flips.shape == (2,)
    assert np.all(updated_flips == [True, False])
    meta = reopened["meta"]
    assert isinstance(meta, FoamFile.SubDict)
    assert meta.as_dict() == {"names": ["first", "empty"]}

    file.format = "ascii"
    file[None] = zones
    ascii_contents = path.read_bytes().decode("ascii")
    assert "List<bool> 2(1 0)" in ascii_contents
    restored = _first_zone(FoamFile(path)[None])
    restored_labels = restored["faceLabels"]
    assert isinstance(restored_labels, np.ndarray)
    assert restored_labels.dtype == np.int64
    assert restored_labels.shape == (2,)
    assert np.all(restored_labels == [1, 2])
    restored_flips = restored["flipMap"]
    assert isinstance(restored_flips, np.ndarray)
    assert restored_flips.dtype == np.bool_
    assert restored_flips.shape == (2,)
    assert np.all(restored_flips == [True, False])


@pytest.mark.parametrize("format_", ["ascii", "binary"])
@pytest.mark.parametrize(
    ("tag", "expected"),
    [("List<label>", [1, 2]), ("List<bool>", [True, False])],
)
def test_generic_list_with_type_token_roundtrip(
    format_: str, tag: str, expected: list[int | bool]
) -> None:
    value: FileDictLike = {
        "FoamFile": {"format": format_},
        "data": ["before", tag, expected, "after"],
    }
    normalized = encoded(value, target=FileDict)
    restored = loads(dumps(value), include_header=True)
    assert isinstance(restored, dict)
    for stage in (normalized, restored):
        data = stage["data"]
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0] == "before"
        assert data[2] == "after"
        values = data[1]
        assert isinstance(values, np.ndarray)
        assert values.dtype == (np.int64 if tag == "List<label>" else np.bool_)
        assert values.shape == (2,)
        assert np.all(values == expected)


def test_zone_keyword_entries_in_list() -> None:
    contents = dumps([("cellLabels", [0, 1]), ("flipMap", [False, True])])
    assert b"cellLabels List<label>" in contents
    assert b"flipMap List<bool>" in contents
    parsed = loads(contents)
    assert isinstance(parsed, list)
    labels, flips = parsed
    assert isinstance(labels, tuple)
    assert isinstance(flips, tuple)
    label_values = labels[1]
    assert isinstance(label_values, np.ndarray)
    assert label_values.dtype == np.int64
    assert label_values.shape == (2,)
    assert np.all(label_values == [0, 1])
    flip_values = flips[1]
    assert isinstance(flip_values, np.ndarray)
    assert flip_values.dtype == np.bool_
    assert flip_values.shape == (2,)
    assert np.all(flip_values == [False, True])


@pytest.mark.parametrize("tag", [b"List<label> ", b""])
def test_binary_labels_preserve_following_entries(tag: bytes) -> None:
    contents = b"cellLabels " + tag + b"1(" + np.int32(7).tobytes() + b");x();"
    parsed = loads(contents)
    assert isinstance(parsed, dict)
    labels = parsed["cellLabels"]
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (1,)
    assert np.all(labels == [7])
    assert parsed["x()"] is None


def test_binary_labels_with_embedded_closing_parenthesis() -> None:
    labels = np.array([7 + (41 << 32)], dtype=np.int64)
    parsed = loads(b"values List<label> 1(" + labels.tobytes() + b");")
    assert isinstance(parsed, dict)
    values = parsed["values"]
    assert isinstance(values, np.ndarray)
    assert values.dtype == labels.dtype
    assert values.shape == labels.shape
    assert np.all(values == labels)


@pytest.mark.parametrize("last_label", [41, 15145, 10537, 32041, 0x2A2F29, 0x2F2F29])
def test_binary_label64_with_embedded_terminators(last_label: int) -> None:
    labels = np.array([1, last_label], dtype=np.int64)
    parsed = loads(b"values List<label> 2(" + labels.tobytes() + b");")
    assert isinstance(parsed, dict)
    assert list(parsed) == ["values"]
    values = parsed["values"]
    assert isinstance(values, np.ndarray)
    assert values.dtype == labels.dtype
    assert values.shape == labels.shape
    assert np.all(values == labels)


def test_invalid_binary_bool() -> None:
    with pytest.raises(FoamFileDecodeError):
        loads(b"values List<bool> 1(\x02);")


def test_standalone_binary_label64_roundtrip() -> None:
    labels = np.array([0, 41, 125], dtype=np.int64)
    parsed = loads(
        b"FoamFile {format binary;} List<label> 3(" + labels.tobytes() + b")",
        include_header=True,
    )
    restored = loads(dumps(parsed))
    assert isinstance(restored, np.ndarray)
    assert restored.dtype == np.int64
    assert restored.shape == labels.shape
    assert np.all(restored == labels)
