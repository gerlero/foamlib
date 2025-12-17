"""Tests for C extension modules."""

import pytest
from foamlib._c._skip import skip
from foamlib._files._parsing.exceptions import FoamFileDecodeError


def test_skip_whitespace() -> None:
    """Test skipping various whitespace characters."""
    assert skip(b"   hello", 0) == 3
    assert skip(b"\t\t\thello", 0) == 3
    assert skip(b"\n\n\nhello", 0) == 3
    assert skip(b" \t\n\r\f\vhello", 0) == 6
    assert skip(b"hello", 0) == 0


def test_skip_line_comment() -> None:
    """Test skipping // comments."""
    assert skip(b"// comment\nhello", 0) == 11
    assert skip(b"//comment\nhello", 0) == 10
    assert skip(b"  // comment\nhello", 0) == 13
    # Line comment with backslash continuation
    assert skip(b"// comment\\\n  more\nhello", 0) == 19


def test_skip_block_comment() -> None:
    """Test skipping /* */ comments."""
    assert skip(b"/* comment */hello", 0) == 13
    assert skip(b"/*comment*/hello", 0) == 11
    assert skip(b"  /* comment */  hello", 0) == 17
    # Multi-line block comment
    assert skip(b"/* comment\nmore\nlines */hello", 0) == 24


def test_skip_mixed_whitespace_and_comments() -> None:
    """Test skipping mixed whitespace and comments."""
    assert skip(b"  /* comment */  // line\nhello", 0) == 25
    assert skip(b"// comment\n  /* block */  hello", 0) == 26


def test_skip_newline_ok_parameter() -> None:
    """Test the newline_ok parameter."""
    # With newline_ok=True (default), newlines are skipped
    assert skip(b"  \n  hello", 0, newline_ok=True) == 5
    # With newline_ok=False, newlines are not skipped
    assert skip(b"  \n  hello", 0, newline_ok=False) == 2
    # Line comments with newline_ok=False
    result = skip(b"// comment\nhello", 0, newline_ok=False)
    assert result == 10  # Stops at the newline


def test_skip_no_content_to_skip() -> None:
    """Test when there's nothing to skip."""
    assert skip(b"hello", 0) == 0
    assert skip(b"123", 0) == 0


def test_skip_at_end_of_buffer() -> None:
    """Test skipping at the end of the buffer."""
    assert skip(b"hello   ", 5) == 8
    assert skip(b"hello", 5) == 5


def test_skip_incomplete_block_comment() -> None:
    """Test that incomplete block comments raise an error."""
    with pytest.raises(FoamFileDecodeError):
        skip(b"/* incomplete comment", 0)


def test_skip_with_bytearray() -> None:
    """Test that the skip function works with bytearray as well as bytes."""
    ba = bytearray(b"  /* comment */  hello")
    assert skip(ba, 0) == 17


def test_skip_multiple_comments() -> None:
    """Test skipping multiple consecutive comments."""
    assert skip(b"/* c1 */ /* c2 */ /* c3 */hello", 0) == 26
    assert skip(b"// c1\n// c2\n// c3\nhello", 0) == 18


def test_skip_empty_comment() -> None:
    """Test skipping empty comments."""
    assert skip(b"/**/hello", 0) == 4
    assert skip(b"//\nhello", 0) == 3
