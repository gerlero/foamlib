from foamlib._files._parsing._re import (
    _FOUR_FACE_LIKE,
    _SYMM_TENSOR,
    _TENSOR,
    _THREE_FACE_LIKE,
    _UNCOMMENTED_FOUR_FACE_LIKE,
    _UNCOMMENTED_SYMM_TENSOR,
    _UNCOMMENTED_TENSOR,
    _UNCOMMENTED_THREE_FACE_LIKE,
    _UNCOMMENTED_VECTOR,
    _VECTOR,
    COMMENT,
    FACES_LIKE_LIST,
    FLOAT,
    FLOAT_LIST,
    INTEGER,
    INTEGER_LIST,
    SKIP,
    SYMM_TENSOR_LIST,
    TENSOR_LIST,
    UNCOMMENTED_FACES_LIKE_LIST,
    UNCOMMENTED_FLOAT_LIST,
    UNCOMMENTED_INTEGER_LIST,
    UNCOMMENTED_SYMM_TENSOR_LIST,
    UNCOMMENTED_TENSOR_LIST,
    UNCOMMENTED_VECTOR_LIST,
    UNSIGNED_INTEGER,
    VECTOR_LIST,
)


def test_comment() -> None:
    assert COMMENT.fullmatch(b"// This is a comment")
    assert COMMENT.fullmatch(b"/* This is a \n multi-line comment */")
    assert not COMMENT.fullmatch(b"This is not a comment")
    assert not COMMENT.fullmatch(b"/* these are *//* two comments */")


def test_whitespace_and_comments() -> None:
    assert SKIP.fullmatch(b"   // Comment\n /* Multi-line \n comment */ \t")
    assert SKIP.fullmatch(b"/* these are *//* two comments */")
    assert not SKIP.fullmatch(b"Not whitespace or comment")


def test_unsigned_integer() -> None:
    assert UNSIGNED_INTEGER.fullmatch(b"12345")
    assert not UNSIGNED_INTEGER.fullmatch(b"-12345")


def test_integer() -> None:
    assert INTEGER.fullmatch(b"-12345")
    assert INTEGER.fullmatch(b"67890")
    assert not INTEGER.fullmatch(b"12.34")


def test_float() -> None:
    assert FLOAT.fullmatch(b"3.14")
    assert FLOAT.fullmatch(b"-2.71e10")
    assert FLOAT.fullmatch(b"NaN")
    assert not FLOAT.fullmatch(b"Not a float")


def test_vector() -> None:
    assert _UNCOMMENTED_VECTOR.fullmatch(b"(1.0 2.0 3.0)")
    assert _UNCOMMENTED_VECTOR.fullmatch(b"(0 0 0)")
    assert _UNCOMMENTED_VECTOR.fullmatch(b"(  -1.5  0  3.14  )")
    assert _UNCOMMENTED_VECTOR.fullmatch(b"(1e3 2.5e-2 -3.0)")
    assert _VECTOR.fullmatch(b"(1.0/*comment*/2.0 3.0)")
    assert not _VECTOR.fullmatch(b"(1.0 2.0)")


def test_symm_tensor() -> None:
    assert _UNCOMMENTED_SYMM_TENSOR.fullmatch(b"(1.0 0.0 0.0 2.0 0.0 3.0)")
    assert _UNCOMMENTED_SYMM_TENSOR.fullmatch(b"(0 0 0 0 0 0)")
    assert _UNCOMMENTED_SYMM_TENSOR.fullmatch(b"(  -1.5  0  3.14  2.71  0  -4.0 )")
    assert _SYMM_TENSOR.fullmatch(b"(1.0/*comment*/0.0 0.0 2.0 0.0 3.0)")
    assert not _SYMM_TENSOR.fullmatch(b"(1.0 2 3.0)")


def test_tensor() -> None:
    assert _UNCOMMENTED_TENSOR.fullmatch(b"(1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0)")
    assert _UNCOMMENTED_TENSOR.fullmatch(b"(0 0 0 0 0 0 0 0 0)")
    assert _UNCOMMENTED_TENSOR.fullmatch(
        b"(  -1.5  0  3.14  0.0  2.71  0  0  0  -4.0 )"
    )
    assert _TENSOR.fullmatch(b"(1.0/*comment*/0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0)")
    assert not _TENSOR.fullmatch(b"(1.0 2.0 3.0)")


def test_integer_list() -> None:
    assert UNCOMMENTED_INTEGER_LIST.fullmatch(b"(1 2 3 4 5)")
    assert INTEGER_LIST.fullmatch(b"(  -1  /*comment*/ 0  3  )")
    assert UNCOMMENTED_INTEGER_LIST.fullmatch(b"()")
    assert not INTEGER_LIST.fullmatch(b"(1.0 2.0)")
    assert not INTEGER_LIST.fullmatch(b"(1, 2)")


def test_float_list() -> None:
    assert UNCOMMENTED_FLOAT_LIST.fullmatch(b"(1.0 2.0 3.0 4.0)")
    assert UNCOMMENTED_FLOAT_LIST.fullmatch(b"(  -1.5  0  3.14  )")
    assert UNCOMMENTED_FLOAT_LIST.fullmatch(b"()")
    assert FLOAT_LIST.fullmatch(b"(1.0/*comment*/2.0 3.0)")
    assert not FLOAT_LIST.fullmatch(b"(1.0, 2.0)")


def test_vector_list() -> None:
    assert UNCOMMENTED_VECTOR_LIST.fullmatch(b"((1.0 2.0 3.0)(4.0 5.0 6.0))")
    assert UNCOMMENTED_VECTOR_LIST.fullmatch(b"( (  -1.5  0  3.14  ) (0 0 0) )")
    assert UNCOMMENTED_VECTOR_LIST.fullmatch(b"()")
    assert VECTOR_LIST.fullmatch(b"((1.0/*comment*/2.0 3.0)(4.0 5.0 6.0))")
    assert not VECTOR_LIST.fullmatch(b"((1.0 2.0),(3.0 4.0))")


def test_symm_tensor_list() -> None:
    assert UNCOMMENTED_SYMM_TENSOR_LIST.fullmatch(
        b"((1.0 0.0 0.0 2.0 0.0 3.0)(4.0 0.0 0.0 5.0 0.0 6.0))"
    )
    assert UNCOMMENTED_SYMM_TENSOR_LIST.fullmatch(
        b"( (1.0 1.0 1.0 1.0 1.0 1.0) (0.0 0.0 0.0 0.0 0.0 0.0) )"
    )
    assert UNCOMMENTED_SYMM_TENSOR_LIST.fullmatch(b"()")
    assert SYMM_TENSOR_LIST.fullmatch(
        b"((1.0/*comment*/0.0 0.0 2.0 0.0 3.0)(4.0 0.0 0.0 5.0 0.0 6.0))"
    )
    assert SYMM_TENSOR_LIST.fullmatch(b"()")
    assert not SYMM_TENSOR_LIST.fullmatch(b"((1.0 2.0 3.0)(4.0 5.0 6.0))")


def test_tensor_list() -> None:
    assert UNCOMMENTED_TENSOR_LIST.fullmatch(
        b"((1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0)(4.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 6.0))"
    )
    assert UNCOMMENTED_TENSOR_LIST.fullmatch(b"()")
    assert TENSOR_LIST.fullmatch(
        b"( (1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0) (0.0 0.0 /* comment */0.0 0.0 0.0 0.0 0.0 0.0 0.0) )"
    )
    assert TENSOR_LIST.fullmatch(
        b"((1.0/*comment*/0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0)(4.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 6.0))"
    )
    assert not TENSOR_LIST.fullmatch(b"((1.0 2.0 3.0)(4.0 5.0 6.0))")


def test_three_face_like() -> None:
    assert _UNCOMMENTED_THREE_FACE_LIKE.fullmatch(b"3(1 2 3)")
    assert _THREE_FACE_LIKE.fullmatch(b"3  (  10 20 30/* comment */  )")
    assert not _THREE_FACE_LIKE.fullmatch(b"(1 2 3)")


def test_four_face_like() -> None:
    assert _UNCOMMENTED_FOUR_FACE_LIKE.fullmatch(b"4(1 2 3 4)")
    assert _FOUR_FACE_LIKE.fullmatch(b"4  (  10 20 30 40 /* comment */ )")
    assert not _FOUR_FACE_LIKE.fullmatch(b"(1 2 3 4)")


def test_faces_like_list() -> None:
    assert UNCOMMENTED_FACES_LIKE_LIST.fullmatch(b"(3(1 2 3)4(4 5 6 7)3(8 9 10))")
    assert UNCOMMENTED_FACES_LIKE_LIST.fullmatch(b"()")
    assert FACES_LIKE_LIST.fullmatch(
        b"(  3 (  1 2 3 ) /* comment */ 4(4 5 6 7) 3(8 9 10)  )"
    )
    assert not FACES_LIKE_LIST.fullmatch(b"(2(1 2)5(1 2 3 4 5 6))")
