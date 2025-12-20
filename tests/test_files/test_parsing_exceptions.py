"""Tests for foamlib._files._parsing.exceptions module - additional coverage."""

from foamlib._files._parsing.exceptions import FoamFileDecodeError


def test_foam_file_decode_error_colno_first_line() -> None:
    """Test FoamFileDecodeError.colno when error is on the first line."""
    contents = b"FoamFile { version 2.0; }"
    pos = 5  # Position at 'i' in "FoamFile"
    
    error = FoamFileDecodeError(contents, pos, expected="something else")
    
    assert error.colno == 6  # pos + 1 since colno is 1-indexed
    assert error.lineno == 1


def test_foam_file_decode_error_end_of_file() -> None:
    """Test FoamFileDecodeError when error is at the end of file without trailing newline."""
    contents = b"FoamFile\n{\n    version 2.0;"
    pos = len(contents) - 1  # Last character
    
    error = FoamFileDecodeError(contents, pos, expected="closing brace")
    
    assert error.lineno == 3
    # The line should be extracted correctly even without trailing newline
    assert "version 2.0;" in error._line


def test_foam_file_decode_error_repr() -> None:
    """Test FoamFileDecodeError.__repr__."""
    contents = b"FoamFile\n{\n    version 2.0;"
    pos = 10
    
    error = FoamFileDecodeError(contents, pos, expected="closing brace")
    
    repr_str = repr(error)
    assert "FoamFileDecodeError" in repr_str
    assert "line=" in repr_str
    assert "column=" in repr_str
