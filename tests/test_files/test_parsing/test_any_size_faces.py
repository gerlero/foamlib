"""Face lists with faces of 5+ vertices take a fast path."""

from typing import Any

import pytest

from foamlib import loads
from foamlib._files._parsing import _parser
from foamlib._files._parsing._parser import ParseError

MIXED = b"""FoamFile { version 2.0; format ascii; class faceList; object faces; }
4
(
4(0 1 2 3)
5(1 2 3 4 5)
6(10 11 12 13 14 15)
3(7 8 9)
)
"""


def test_same_result_as_the_generic_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    fast = loads(MIXED)

    def unavailable(*_: Any) -> Any:
        raise ParseError(b"", 0, expected="")

    monkeypatch.setattr(_parser, "_parse_ascii_any_size_faces_list", unavailable)
    generic = loads(MIXED)

    assert (
        fast
        == generic
        == [[0, 1, 2, 3], [1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15], [7, 8, 9]]
    )
    assert all(type(v) is int for face in fast for v in face)  # ty: ignore[not-iterable]


@pytest.mark.parametrize(
    "faces",
    [
        b"2(4(0 1 2 3) 5(1 2 3 4))",  # a face shorter than its size
        b"2(4(0 1 2 3) 5(1 2 3 4 5 6))",  # a face longer than its size
        b"3(4(0 1 2 3) 5(1 2 3 4 5))",  # fewer faces than the count
    ],
)
def test_inconsistent_sizes_are_rejected(faces: bytes) -> None:
    with pytest.raises(ParseError):
        _parser._parse_ascii_any_size_faces_list(faces, 0)


def test_generic_parser_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    parse_list = _parser._parse_list

    def counting(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return parse_list(*args)

    monkeypatch.setattr(_parser, "_parse_list", counting)
    n = 1000
    faces = " ".join(f"{4 + i % 3}({' '.join(['1'] * (4 + i % 3))})" for i in range(n))
    assert len(loads(f"{n}({faces})")) == n  # ty: ignore[invalid-argument-type]
    assert calls <= 1  # only the check for another entry after the list


@pytest.mark.parametrize(
    "faces",
    [
        b"3(4(0 1 2 3) 3(7 8 9) 4(4 5 6 7))",
        b"(\n3 ( 1 2 3 )\n4(0 1\n2 3)\n)",
        b"0()",
    ],
)
def test_faces_of_3_and_4_vertices_give_arrays(faces: bytes) -> None:
    fast, end = _parser._parse_ascii_any_size_faces_list(faces, 0)
    reference, reference_end = _parser._parse_ascii_faces_like_list(faces, 0)

    assert end == reference_end
    assert len(fast) == len(reference)
    for face, reference_face in zip(fast, reference, strict=True):
        assert face.dtype == reference_face.dtype
        assert face.tolist() == reference_face.tolist()


def test_zero_padded_sizes_are_left_to_other_parsers() -> None:
    with pytest.raises(ParseError):
        _parser._parse_ascii_any_size_faces_list(b"1(03(1 2 3))", 0)
    assert loads("1(03(1 2 3))") == [[1, 2, 3]]
