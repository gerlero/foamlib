import re

import regex

TOKEN = regex.compile(
    rb'"(?:[^"\\]|\\.)*"|[A-Za-z_#$][\!-\'\*-\:<-Z\^-z\|]*(\((?:[\!-\'\*-\:<-Z\^-z\|]*|(?1))*\))?'
)
UNSIGNED_INTEGER = re.compile(rb"\d++(?![\.a-zA-Z_])")
INTEGER = re.compile(rb"[+-]?\d++(?![\.a-zA-Z_])")
FLOAT = re.compile(
    rb"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan(?![A-Za-z0-9_])|inf(?:inity)?(?![A-Za-z0-9_])))"
)

COMMENT = re.compile(rb"(?:/\*(?:[^*]|\*(?!/))*\*/)|(?://(?:\\\n|[^\n])*)")
SKIP = re.compile(rb"(?:\s+|(?:" + COMMENT.pattern + b"))+")
SKIP_NO_NEWLINE = re.compile(rb"(?:[ \t\r]+|(?:" + COMMENT.pattern + b"))+")

_VECTOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + FLOAT.pattern
    + rb"(?:"
    + SKIP.pattern
    + rb")){2}(?:"
    + FLOAT.pattern
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_VECTOR = re.compile(
    rb"\((?:\s*)?(?:"
    + FLOAT.pattern
    + rb"(?:\s*)){2}(?:"
    + FLOAT.pattern
    + rb")(?:\s*)?\)"
)
_SYMM_TENSOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + FLOAT.pattern
    + rb"(?:"
    + SKIP.pattern
    + rb")){5}(?:"
    + FLOAT.pattern
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_SYMM_TENSOR = re.compile(
    rb"\((?:\s*)?(?:"
    + FLOAT.pattern
    + rb"(?:\s*)){5}(?:"
    + FLOAT.pattern
    + rb")(?:\s*)?\)"
)
_TENSOR = re.compile(
    rb"\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + FLOAT.pattern
    + rb"(?:"
    + SKIP.pattern
    + rb")){8}(?:"
    + FLOAT.pattern
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_TENSOR = re.compile(
    rb"\((?:\s*)?(?:"
    + FLOAT.pattern
    + rb"(?:\s*)){8}(?:"
    + FLOAT.pattern
    + rb")(?:\s*)?\)"
)

INTEGER_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + INTEGER.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_INTEGER_LIST = re.compile(
    rb"\((?:(?:\s*)?(?:" + INTEGER.pattern + rb"))*(?:\s*)?\)"
)

FLOAT_LIST = re.compile(
    rb"\((?:(?:"
    + SKIP.pattern
    + rb")?(?:"
    + FLOAT.pattern
    + rb"))*(?:"
    + SKIP.pattern
    + rb")?\)"
)
UNCOMMENTED_FLOAT_LIST = re.compile(
    rb"\((?:(?:\s*)?(?:" + FLOAT.pattern + rb"))*(?:\s*)?\)"
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
    rb"\((?:(?:\s*)?(?:" + _UNCOMMENTED_VECTOR.pattern + rb"))*(?:\s*)?\)"
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
    rb"\((?:(?:\s*)?(?:" + _UNCOMMENTED_SYMM_TENSOR.pattern + rb"))*(?:\s*)?\)"
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
    rb"\((?:(?:\s*)?(?:" + _UNCOMMENTED_TENSOR.pattern + rb"))*(?:\s*)?\)"
)

_THREE_FACE_LIKE = re.compile(
    rb"3(?:"
    + SKIP.pattern
    + rb")?\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + INTEGER.pattern
    + rb"(?:"
    + SKIP.pattern
    + rb")){2}(?:"
    + INTEGER.pattern
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_THREE_FACE_LIKE = re.compile(
    rb"3(?:\s*)?\((?:\s*)?(?:"
    + INTEGER.pattern
    + rb"(?:\s*)){2}(?:"
    + INTEGER.pattern
    + rb")(?:\s*)?\)"
)
_FOUR_FACE_LIKE = re.compile(
    rb"4(?:"
    + SKIP.pattern
    + rb")?\((?:"
    + SKIP.pattern
    + rb")?(?:"
    + INTEGER.pattern
    + rb"(?:"
    + SKIP.pattern
    + rb")){3}(?:"
    + INTEGER.pattern
    + rb")(?:"
    + SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_FOUR_FACE_LIKE = re.compile(
    rb"4(?:\s*)?\((?:\s*)?(?:"
    + INTEGER.pattern
    + rb"(?:\s*)){3}(?:"
    + INTEGER.pattern
    + rb")(?:\s*)?\)"
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
    rb"\((?:(?:\s*)?(?:"
    + _UNCOMMENTED_THREE_FACE_LIKE.pattern
    + rb"|"
    + _UNCOMMENTED_FOUR_FACE_LIKE.pattern
    + rb"))*(?:\s*)?\)"
)
