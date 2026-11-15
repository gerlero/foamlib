"""Text that is as long as a binary list must be read as text."""

from typing import Any

from foamlib import loads


def test_text_as_long_as_a_binary_list_is_read_as_text() -> None:
    # These are as long as binary lists of 2 int32, 1 float64 and 1 vector
    assert loads(b"2(abc defg)") == ["abc", "defg"]
    assert loads(b"1(2(10 20))") == [[10, 20]]
    assert loads(b"1(\n6  ( 7 345 12 0 345 12))") == [[7, 345, 12, 0, 345, 12]]


def test_binary_lists_are_still_read() -> None:
    binary: Any = loads(b"2(\x07\x00\x00\x00\x08\x00\x00\x00)")
    assert binary.tolist() == [7, 8]
