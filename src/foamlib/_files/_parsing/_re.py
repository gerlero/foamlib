"""Optimized regular expression patterns for OpenFOAM file parsing.

Patterns are generated at import time for improved readability and matching performance.
Optimizations focus on reducing non-capturing groups and simplifying pattern structures.
"""
import re

# Basic numeric patterns
UNSIGNED_INTEGER = re.compile(rb"\d+", re.ASCII)
INTEGER = re.compile(rb"[+-]?\d+", re.ASCII)

# Optimized FLOAT pattern - minimized nested groups for better matching performance
# Matches: 123, -45.6, 7.8e-9, nan, NaN, inf, Infinity, etc.
# Using (?i:...) for case-insensitive matching is faster than explicit [Nn][Aa][Nn]
_FLOAT_NUM = rb"\d+\.?\d*(?:e[+-]?\d+)?"
_FLOAT_SPECIAL = rb"nan|inf(?:inity)?"
_FLOAT_PATTERN = rb"(?i:[+-]?(?:" + _FLOAT_NUM + rb"|" + _FLOAT_SPECIAL + rb"))"
FLOAT = re.compile(_FLOAT_PATTERN, re.ASCII)

# Comment patterns - optimized for matching performance
_COMMENT_BLOCK = rb"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/"  # More efficient than negative lookahead
_COMMENT_LINE = rb"//(?:\\\n|[^\n])*"
_COMMENT_PATTERN = rb"(?:" + _COMMENT_BLOCK + rb"|" + _COMMENT_LINE + rb")"
COMMENT = re.compile(_COMMENT_PATTERN)

# Skip patterns (whitespace and comments) - simplified structure
_SKIP_PATTERN = rb"(?:\s+|" + _COMMENT_BLOCK + rb"|" + _COMMENT_LINE + rb")+"
SKIP = re.compile(_SKIP_PATTERN)
SKIP_NO_NEWLINE = re.compile(
    rb"(?:[ \t\r]+|" + _COMMENT_BLOCK + rb"|" + _COMMENT_LINE + rb")+"
)

# Helper to build numeric tuple patterns (vectors, tensors)
# Using string templates with format for clarity
def _make_tuple_pattern(n: int, elem_pattern: bytes, skip_pattern: bytes) -> bytes:
    """Generate pattern for n-element tuple with optional skip between elements."""
    if n == 0:
        return rb"\(\s*\)"
    # First element with optional leading skip
    parts = [rb"\((?:", skip_pattern, rb")?", elem_pattern]
    # Middle elements with required skip before each
    for _ in range(n - 1):
        parts.extend([skip_pattern, elem_pattern])
    # Optional trailing skip and closing paren
    parts.extend([rb"(?:", skip_pattern, rb")?\)"])
    return b"".join(parts)


def _make_uncommented_tuple_pattern(n: int, elem_pattern: bytes) -> bytes:
    """Generate pattern for n-element tuple with only whitespace."""
    if n == 0:
        return rb"\(\s*\)"
    parts = [rb"\(\s*", elem_pattern]
    for _ in range(n - 1):
        parts.extend([rb"\s*", elem_pattern])
    parts.append(rb"\s*\)")
    return b"".join(parts)


# Vector, tensor patterns (3, 6, 9 floats)
_VECTOR = re.compile(_make_tuple_pattern(3, _FLOAT_PATTERN, _SKIP_PATTERN))
_UNCOMMENTED_VECTOR = re.compile(
    _make_uncommented_tuple_pattern(3, _FLOAT_PATTERN), re.ASCII
)

_SYMM_TENSOR = re.compile(_make_tuple_pattern(6, _FLOAT_PATTERN, _SKIP_PATTERN))
_UNCOMMENTED_SYMM_TENSOR = re.compile(
    _make_uncommented_tuple_pattern(6, _FLOAT_PATTERN), re.ASCII
)

_TENSOR = re.compile(_make_tuple_pattern(9, _FLOAT_PATTERN, _SKIP_PATTERN))
_UNCOMMENTED_TENSOR = re.compile(
    _make_uncommented_tuple_pattern(9, _FLOAT_PATTERN), re.ASCII
)

# Face-like patterns (3 or 4 integers)
_THREE_FACE_LIKE = re.compile(
    rb"3(?:" + _SKIP_PATTERN + rb")?" + _make_tuple_pattern(3, INTEGER.pattern, _SKIP_PATTERN)
)
_UNCOMMENTED_THREE_FACE_LIKE = re.compile(
    rb"3" + _make_uncommented_tuple_pattern(3, INTEGER.pattern), re.ASCII
)

_FOUR_FACE_LIKE = re.compile(
    rb"4(?:" + _SKIP_PATTERN + rb")?" + _make_tuple_pattern(4, INTEGER.pattern, _SKIP_PATTERN)
)
_UNCOMMENTED_FOUR_FACE_LIKE = re.compile(
    rb"4" + _make_uncommented_tuple_pattern(4, INTEGER.pattern), re.ASCII
)


# List patterns - lists of elements
def _make_list_pattern(elem_pattern: bytes, skip_pattern: bytes) -> bytes:
    """Generate pattern for list of elements with optional skip."""
    return (
        rb"\((?:(?:" + skip_pattern + rb")?(?:" + elem_pattern + rb"))*(?:"
        + skip_pattern + rb")?\)"
    )


def _make_uncommented_list_pattern(elem_pattern: bytes) -> bytes:
    """Generate pattern for list of elements with only whitespace."""
    return rb"\((?:\s*(?:" + elem_pattern + rb"))*\s*\)"


INTEGER_LIST = re.compile(_make_list_pattern(INTEGER.pattern, _SKIP_PATTERN))
UNCOMMENTED_INTEGER_LIST = re.compile(
    _make_uncommented_list_pattern(INTEGER.pattern), re.ASCII
)

FLOAT_LIST = re.compile(_make_list_pattern(_FLOAT_PATTERN, _SKIP_PATTERN))
UNCOMMENTED_FLOAT_LIST = re.compile(
    _make_uncommented_list_pattern(_FLOAT_PATTERN), re.ASCII
)

VECTOR_LIST = re.compile(_make_list_pattern(_VECTOR.pattern, _SKIP_PATTERN))
UNCOMMENTED_VECTOR_LIST = re.compile(
    _make_uncommented_list_pattern(_UNCOMMENTED_VECTOR.pattern), re.ASCII
)

SYMM_TENSOR_LIST = re.compile(_make_list_pattern(_SYMM_TENSOR.pattern, _SKIP_PATTERN))
UNCOMMENTED_SYMM_TENSOR_LIST = re.compile(
    _make_uncommented_list_pattern(_UNCOMMENTED_SYMM_TENSOR.pattern), re.ASCII
)

TENSOR_LIST = re.compile(_make_list_pattern(_TENSOR.pattern, _SKIP_PATTERN))
UNCOMMENTED_TENSOR_LIST = re.compile(
    _make_uncommented_list_pattern(_UNCOMMENTED_TENSOR.pattern), re.ASCII
)

_FACES_PATTERN = rb"(?:" + _THREE_FACE_LIKE.pattern + rb"|" + _FOUR_FACE_LIKE.pattern + rb")"
_UNCOMMENTED_FACES_PATTERN = (
    rb"(?:" + _UNCOMMENTED_THREE_FACE_LIKE.pattern + rb"|"
    + _UNCOMMENTED_FOUR_FACE_LIKE.pattern + rb")"
)

FACES_LIKE_LIST = re.compile(_make_list_pattern(_FACES_PATTERN, _SKIP_PATTERN))
UNCOMMENTED_FACES_LIKE_LIST = re.compile(
    _make_uncommented_list_pattern(_UNCOMMENTED_FACES_PATTERN), re.ASCII
)
