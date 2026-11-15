"""The patterns for numeric lists and comments must not backtrack."""

import time

from foamlib import loads


def test_list_that_is_not_numeric_is_rejected_quickly() -> None:
    # Rejecting it as a numeric list took 8 times longer per extra number
    start = time.perf_counter()
    assert loads(b"(" + b"5100 " * 8 + b"abc)") == [5100] * 8 + ["abc"]
    assert time.perf_counter() - start < 1


def test_comment_text_is_not_read_as_data() -> None:
    assert loads(b"(1 2 // 3 )\n foo)") == [1, 2, "foo"]
    assert loads(b"(// see (0)\n foo)") == ["foo"]


def test_digits_of_a_number_are_not_split_into_a_vector() -> None:
    assert loads(b"((123) (456) (789))") == [[123], [456], [789]]
