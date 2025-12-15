import re

UNSIGNED_INTEGER = re.compile(rb"\d+", re.ASCII)
INTEGER = re.compile(rb"[+-]?\d+", re.ASCII)
FLOAT = re.compile(
    rb"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))", re.ASCII
)

# Simplified patterns for list parsing: match "number-like tokens" without strict validation
# These patterns only validate structure, letting numpy.fromstring handle actual numeric validation
#
# Design philosophy:
# - Be permissive enough to match all valid numbers (including edge cases)
# - Allow some invalid patterns (e.g., "123.", "1e+") to pass regex but fail at numpy stage
# - This is intentional: numpy.fromstring will catch these and we convert ValueError to ParseError
# - Performance benefit: simpler regex patterns execute faster than complex strict patterns
#
# For floats: match sequences that look like floats - simpler than original FLOAT pattern
# Matches: signed numbers with optional decimal/exponent, or NaN/Inf keywords (case-insensitive)
# Examples that match: "1", "1.0", "1.", ".5", "1e5", "1.2e-3", "NaN", "Inf", "Infinity"
# Examples that match but will fail at numpy: "1e+", "1.2.3", "..5"  (these are caught later)
_FLOAT_LIKE = rb"[+-]?(?:[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?|[0-9]*\.[0-9]+(?:[eE][+-]?[0-9]+)?|[Nn][Aa][Nn]|[Ii][Nn][Ff](?:[Ii][Nn][Ii][Tt][Yy])?)"
# For integers: match sequences that look like integers (digits with optional sign, no decimal)
_INTEGER_LIKE = rb"[+-]?[0-9]+"

COMMENT = re.compile(rb"(?:/\*(?:[^*]|\*(?!/))*\*/)|(?://(?:\\\n|[^\n])*)")
SKIP = re.compile(rb"(?:\s+|(?:" + COMMENT.pattern + b"))+")
SKIP_NO_NEWLINE = re.compile(rb"(?:[ \t\r]+|(?:" + COMMENT.pattern + b"))+")

_VECTOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_VECTOR = re.compile(
    rb"\(\s*(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb")\s*\)",
    re.ASCII,
)
_SYMM_TENSOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_SYMM_TENSOR = re.compile(
    rb"\(\s*(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb")\s*\)",
    re.ASCII,
)
_TENSOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _FLOAT_LIKE
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_TENSOR = re.compile(
    rb"\(\s*(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb"\s*)(?:"
    + _FLOAT_LIKE
    + rb")\s*\)",
    re.ASCII,
)

INTEGER_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _INTEGER_LIKE
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_INTEGER_LIST = re.compile(rb"\((?:\s*(?:" + _INTEGER_LIKE + rb"))*\s*\)")

FLOAT_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _FLOAT_LIKE
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_FLOAT_LIST = re.compile(
    rb"\((?:\s*(?:" + _FLOAT_LIKE + rb"))*\s*\)", re.ASCII
)
VECTOR_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _VECTOR.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_VECTOR_LIST = re.compile(
    rb"\((?:\s*(?:" + _UNCOMMENTED_VECTOR.pattern + rb"))*\s*\)", re.ASCII
)
SYMM_TENSOR_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _SYMM_TENSOR.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_SYMM_TENSOR_LIST = re.compile(
    rb"\((?:\s*(?:" + _UNCOMMENTED_SYMM_TENSOR.pattern + rb"))*\s*\)", re.ASCII
)
TENSOR_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _TENSOR.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_TENSOR_LIST = re.compile(
    rb"\((?:\s*(?:" + _UNCOMMENTED_TENSOR.pattern + rb"))*\s*\)", re.ASCII
)

_THREE_FACE_LIKE = re.compile(
    rb"3(?:"
    + SKIP.pattern
    + rb")?\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + _INTEGER_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _INTEGER_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _INTEGER_LIKE
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_THREE_FACE_LIKE = re.compile(
    rb"3\s*\(\s*(?:"
    + _INTEGER_LIKE
    + rb"\s*)(?:"
    + _INTEGER_LIKE
    + rb"\s*)(?:"
    + _INTEGER_LIKE
    + rb")\s*\)",
    re.ASCII,
)
_FOUR_FACE_LIKE = re.compile(
    rb"4(?:"
    + SKIP.pattern
    + rb")?\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + _INTEGER_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _INTEGER_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _INTEGER_LIKE
    + rb"(?:"
    + SKIP.pattern
    + rb"))(?:"
    + _INTEGER_LIKE
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_FOUR_FACE_LIKE = re.compile(
    rb"4\s*\(\s*(?:"
    + _INTEGER_LIKE
    + rb"\s*)(?:"
    + _INTEGER_LIKE
    + rb"\s*)(?:"
    + _INTEGER_LIKE
    + rb"\s*)(?:"
    + _INTEGER_LIKE
    + rb")\s*\)",
    re.ASCII,
)
FACES_LIKE_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + _THREE_FACE_LIKE.pattern
    + rb"|"
    + _FOUR_FACE_LIKE.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_FACES_LIKE_LIST = re.compile(
    rb"\((?:\s*(?:"
    + _UNCOMMENTED_THREE_FACE_LIKE.pattern
    + rb"|"
    + _UNCOMMENTED_FOUR_FACE_LIKE.pattern
    + rb"))*\s*\)",
    re.ASCII,
)
