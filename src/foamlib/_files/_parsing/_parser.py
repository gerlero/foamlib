import contextlib
import dataclasses
import re
import sys
from types import EllipsisType
from typing import Generic, Literal, TypeVar, overload

from foamlib._files._util import add_to_mapping

if sys.version_info >= (3, 11):
    from typing import Unpack, assert_never
else:
    from typing_extensions import Unpack, assert_never

import numpy as np
from multicollections import MultiDict

from ...typing import (
    Data,
    DataEntry,
    Dict,
    Dimensioned,
    DimensionSet,
    Field,
    FileDict,
    KeywordEntry,
    StandaloneData,
    StandaloneDataEntry,
    SubDict,
    Tensor,
)
from .exceptions import FoamFileDecodeError

_DType = TypeVar("_DType", float, int)
_NumpyDType = TypeVar("_NumpyDType", np.float64, np.float32, np.int64, np.int32)
_ElShape = TypeVar(
    "_ElShape", tuple[()], tuple[Literal[3]], tuple[Literal[6]], tuple[Literal[9]]
)
_Output = TypeVar("_Output", FileDict, Data, StandaloneData, str)

_IS_TOKEN_START = [False] * 256
for c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_#$":
    _IS_TOKEN_START[c] = True

_IS_TOKEN_CONTINUATION = _IS_TOKEN_START.copy()
for c in b"0123456789._<>#$:+-*/|^%&=!":
    _IS_TOKEN_CONTINUATION[c] = True

_IS_WHITESPACE = [False] * 256
for c in b" \n\t\r\f\v":
    _IS_WHITESPACE[c] = True

_IS_WHITESPACE_NO_NEWLINE = _IS_WHITESPACE.copy()
_IS_WHITESPACE_NO_NEWLINE[ord(b"\n")] = False

_IS_POSSIBLE_FLOAT = [False] * 256
for c in b"0123456789.-+eEinfnatyINFNATY":
    _IS_POSSIBLE_FLOAT[c] = True

_IS_POSSIBLE_INTEGER = [False] * 256
for c in b"0123456789-+":
    _IS_POSSIBLE_INTEGER[c] = True

_COMMENTS = re.compile(rb"(?:(?:/\*(?:[^*]|\*(?!/))*\*/)|(?://(?:\\\n|[^\n])*))+")
_SKIP = re.compile(rb"(?:\s+|" + _COMMENTS.pattern + rb")+")
_POSSIBLE_FLOAT = re.compile(rb"[0-9.\-+einfatyEINFATY]+", re.ASCII)
_POSSIBLE_INTEGER = re.compile(rb"[0-9\-+]+", re.ASCII)


class ParseError(Exception):
    def __init__(self, contents: bytes | bytearray, pos: int, *, expected: str) -> None:
        self._contents = contents
        self.pos = pos
        self._expected = expected
        super().__init__()

    def make_fatal(self) -> FoamFileDecodeError:
        return FoamFileDecodeError(self._contents, self.pos, expected=self._expected)


def _skip(
    contents: bytes | bytearray,
    pos: int,
    *,
    newline_ok: bool = True,
) -> int:
    is_whitespace = _IS_WHITESPACE if newline_ok else _IS_WHITESPACE_NO_NEWLINE

    with contextlib.suppress(IndexError):
        while True:
            while is_whitespace[contents[pos]]:
                pos += 1

            next1 = contents[pos]
            next2 = contents[pos + 1]

            if next1 == ord("/") and next2 == ord("/"):
                pos += 2
                while True:
                    if contents[pos] == ord("\n"):
                        if newline_ok:
                            pos += 1
                        break
                    if contents[pos] == ord("\\"):
                        with contextlib.suppress(IndexError):
                            if contents[pos + 1] == ord("\n"):
                                pos += 1
                        pos += 1
                        continue
                    pos += 1
                continue

            if next1 == ord("/") and next2 == ord("*"):
                if (pos := contents.find(b"*/", pos + 2)) == -1:
                    raise FoamFileDecodeError(
                        contents,
                        len(contents),
                        expected="*/",
                    )
                pos += 2
                continue
            break

    return pos


def _expect(contents: bytes | bytearray, pos: int, expected: bytes | bytearray) -> int:
    length = len(expected)
    if contents[pos : pos + length] != expected:
        raise ParseError(contents, pos, expected=repr(expected.decode("ascii")))

    return pos + length


def _parse_token(contents: bytes | bytearray, pos: int) -> tuple[str, int]:
    try:
        first = contents[pos]
    except IndexError:
        raise ParseError(contents, pos, expected="token") from None
    try:
        second = contents[pos + 1]
    except IndexError:
        second = None

    if (first_ok := _IS_TOKEN_START[first]) and second is None:
        return contents[pos : pos + 1].decode("ascii"), pos + 1

    if first_ok and (first != ord(b"#") or second != ord(b"{")):
        end = pos + 1
        depth = 0
        char: int = second  # ty: ignore[invalid-assignment]
        with contextlib.suppress(IndexError):
            while True:
                if (depth == 0 and _IS_TOKEN_CONTINUATION[char]) or (
                    depth > 0 and char not in b";(){}[]"
                ):
                    end += 1
                elif char == ord(b"("):
                    depth += 1
                    end += 1
                elif char == ord(b")") and depth > 0:
                    depth -= 1
                    end += 1
                else:
                    break
                char = contents[end]

        if depth != 0:
            raise FoamFileDecodeError(contents, pos, expected=")")

        return contents[pos:end].decode("ascii"), end

    if first == ord(b'"'):
        end = pos + 1
        with contextlib.suppress(IndexError):
            while True:
                char = contents[end]
                if char == ord(b"\\"):
                    end += 2
                elif char == ord(b'"'):
                    return contents[pos : end + 1].decode(), end + 1
                else:
                    end += 1

        raise FoamFileDecodeError(contents, pos, expected="end of quoted string")

    if first == ord(b"#") and second == ord(b"{"):
        if (end := contents.find(b"#}", pos + 2)) == -1:
            raise FoamFileDecodeError(contents, len(contents), expected="#}")
        return contents[pos : end + 2].decode(), end + 2

    raise ParseError(contents, pos, expected="token")


@overload
def _parse_number(
    contents: bytes | bytearray, pos: int, *, target: type[int] = ...
) -> tuple[int, int]: ...


@overload
def _parse_number(
    contents: bytes | bytearray, pos: int, *, target: type[float] = ...
) -> tuple[float, int]: ...


@overload
def _parse_number(
    contents: bytes | bytearray, pos: int, *, target: type[int | float] = ...
) -> tuple[int | float, int]: ...


def _parse_number(
    contents: bytes | bytearray,
    pos: int,
    *,
    target: type[int] | type[float] | type[int | float] = int | float,
) -> tuple[int | float, int]:
    is_numeric = _IS_POSSIBLE_INTEGER if target is int else _IS_POSSIBLE_FLOAT
    end = pos
    with contextlib.suppress(IndexError):
        while is_numeric[contents[end]]:
            end += 1

        if _IS_TOKEN_CONTINUATION[contents[end]]:
            raise ParseError(contents, pos, expected="number")

    if pos == end:
        raise ParseError(contents, pos, expected="number")

    chars = contents[pos:end]
    if target is not float:
        try:
            return int(chars), end
        except ValueError as e:
            if target is int:
                raise ParseError(contents, pos, expected="integer") from e
    try:
        return float(chars), end
    except ValueError as e:
        if target is float:
            raise ParseError(contents, pos, expected="float") from e
        raise ParseError(contents, pos, expected="number") from e


class _ASCIINumericListParser(Generic[_DType, _ElShape]):
    def __init__(self, *, dtype: type[_DType], elshape: _ElShape) -> None:
        self._dtype = dtype
        self._elshape = elshape

        if not elshape:
            self._pattern = re.compile(
                rb"(?:(?:"
                + _SKIP.pattern
                + rb")?(?:"
                + _POSSIBLE_FLOAT.pattern
                + rb"))*(?:"
                + _SKIP.pattern
                + rb")?\)"
            )
            self._uncommented_pattern = re.compile(
                rb"(?:\s*(?:" + _POSSIBLE_FLOAT.pattern + rb"))*\s*\)", re.ASCII
            )
        else:
            (dim,) = elshape
            self._pattern = re.compile(
                rb"(?:(?:"
                + _SKIP.pattern
                + rb")?\("
                + (rb"(?:" + _POSSIBLE_FLOAT.pattern + rb")" + _SKIP.pattern) * dim
                + rb"\))*(?:"
                + _SKIP.pattern
                + rb")?\)"
            )
            self._uncommented_pattern = re.compile(
                rb"(?:\s*\("
                + (rb"(?:" + _POSSIBLE_FLOAT.pattern + rb")\s*") * dim
                + rb"\)\s*)*\)",
                re.ASCII,
            )

    @overload
    def __call__(
        self: "_ASCIINumericListParser[float, _ElShape]",
        contents: bytes | bytearray,
        pos: int,
        *,
        empty_ok: bool = ...,
    ) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.float64]], int]: ...

    @overload
    def __call__(
        self: "_ASCIINumericListParser[int, _ElShape]",
        contents: bytes | bytearray,
        pos: int,
        *,
        empty_ok: bool = ...,
    ) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.int64]], int]: ...

    def __call__(
        self,
        contents: bytes | bytearray,
        pos: int,
        *,
        empty_ok: bool = False,
    ) -> tuple[
        np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.float64 | np.int64]], int
    ]:
        try:
            count, pos = _parse_number(contents, pos, target=int)
        except ParseError:
            count = None
        else:
            if count < 0:
                raise ParseError(contents, pos, expected="non-negative list count")
            if count == 0 and not empty_ok:
                raise ParseError(contents, pos, expected="non-empty numeric list")
            pos = _skip(contents, pos)

        if count is not None and contents[pos : pos + 1] == b"{":
            pos += 1
            pos = _skip(contents, pos)

            if not self._elshape:
                value, pos = _parse_number(contents, pos, target=self._dtype)
                pos = _expect(contents, pos, b"}")
                return np.full((count,), value, dtype=self._dtype), pos

            pos = _expect(contents, pos, b"(")
            values = []
            for _ in range(self._elshape[0]):
                pos = _skip(contents, pos)
                value, pos = _parse_number(contents, pos, target=self._dtype)
                values.append(value)
            pos = _skip(contents, pos)
            pos = _expect(contents, pos, b")")
            pos = _expect(contents, pos, b"}")
            return np.full((count, *self._elshape), values, dtype=self._dtype), pos

        pos = _expect(contents, pos, b"(")
        if match := self._uncommented_pattern.match(contents, pos):
            data = contents[pos : match.end() - 1]
            pos = match.end()

        elif match := self._pattern.match(contents, pos):
            data = contents[pos : match.end() - 1]
            pos = match.end()
            data = _COMMENTS.sub(b" ", data)

        if not match:
            raise ParseError(
                contents,
                pos,
                expected=f"numeric list of type {self._dtype} and shape {self._elshape}",
            )

        if self._elshape:
            data = data.replace(b"(", b" ").replace(b")", b" ")

        data = data.decode("ascii")

        try:
            ret = np.fromstring(data, sep=" ", dtype=self._dtype)
        except ValueError as e:
            raise ParseError(
                contents,
                pos,
                expected=f"numeric list of type {self._dtype} and shape {self._elshape}",
            ) from e

        if self._elshape:
            ret = ret.reshape((-1, *self._elshape))

        if count is None:
            if not empty_ok and len(ret) == 0:
                raise ParseError(contents, pos, expected="non-empty numeric list")
        elif len(ret) != count:
            raise ParseError(
                contents, pos, expected=f"{count} elements (got {len(ret)})"
            )

        return ret, pos


_parse_ascii_integer_list = _ASCIINumericListParser(dtype=int, elshape=())
_parse_ascii_float_list = _ASCIINumericListParser(dtype=float, elshape=())
_parse_ascii_vector_list = _ASCIINumericListParser(dtype=float, elshape=(3,))
_parse_ascii_symm_tensor_list = _ASCIINumericListParser(dtype=float, elshape=(6,))
_parse_ascii_tensor_list = _ASCIINumericListParser(dtype=float, elshape=(9,))


_THREE_FACE_LIKE = re.compile(
    rb"3(?:"
    + _SKIP.pattern
    + rb")?\((?:"
    + _SKIP.pattern
    + rb")?(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"(?:"
    + _SKIP.pattern
    + rb"))(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"(?:"
    + _SKIP.pattern
    + rb"))(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb")(?:"
    + _SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_THREE_FACE_LIKE = re.compile(
    rb"3\s*\(\s*(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"\s*)(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"\s*)(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb")\s*\)",
    re.ASCII,
)
_FOUR_FACE_LIKE = re.compile(
    rb"4(?:"
    + _SKIP.pattern
    + rb")?\((?:"
    + _SKIP.pattern
    + rb")?(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"(?:"
    + _SKIP.pattern
    + rb"))(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"(?:"
    + _SKIP.pattern
    + rb"))(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"(?:"
    + _SKIP.pattern
    + rb"))(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb")(?:"
    + _SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_FOUR_FACE_LIKE = re.compile(
    rb"4\s*\(\s*(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"\s*)(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"\s*)(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb"\s*)(?:"
    + _POSSIBLE_INTEGER.pattern
    + rb")\s*\)",
    re.ASCII,
)
_FACES_LIKE_LIST = re.compile(
    rb"(?:(?:"
    + _SKIP.pattern
    + rb")?(?:"
    + _THREE_FACE_LIKE.pattern
    + rb"|"
    + _FOUR_FACE_LIKE.pattern
    + rb"))*(?:"
    + _SKIP.pattern
    + rb")?\)"
)
_UNCOMMENTED_FACES_LIKE_LIST = re.compile(
    rb"(?:\s*(?:"
    + _UNCOMMENTED_THREE_FACE_LIKE.pattern
    + rb"|"
    + _UNCOMMENTED_FOUR_FACE_LIKE.pattern
    + rb"))*\s*\)",
    re.ASCII,
)


def _parse_ascii_faces_like_list(
    contents: bytes | bytearray, pos: int
) -> tuple[list[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]]], int]:
    try:
        count, pos = _parse_number(contents, pos, target=int)
    except ParseError:
        count = None
    else:
        if count < 0:
            raise ParseError(contents, pos, expected="non-negative list count")
        pos = _skip(contents, pos)

    pos = _expect(contents, pos, b"(")

    if match := _UNCOMMENTED_FACES_LIKE_LIST.match(contents, pos):
        data = contents[pos : match.end() - 1]
        pos = match.end()

    elif match := _FACES_LIKE_LIST.match(contents, pos):
        data = contents[pos : match.end() - 1]
        pos = match.end()

        data = _COMMENTS.sub(b" ", data)

    if not match:
        raise ParseError(contents, pos, expected="faces-like list")

    data = data.replace(b"(", b" ").replace(b")", b" ")
    data = data.decode("ascii")

    try:
        values = np.fromstring(data, sep=" ", dtype=int)
    except ValueError as e:
        raise ParseError(contents, pos, expected="faces-like list") from e

    ret: list[np.ndarray] = []
    i = 0
    while i < len(values):
        n = values[i]
        ret.append(values[i + 1 : i + n + 1])
        i += n + 1

    if count is not None and len(ret) != count:
        raise ParseError(contents, pos, expected=f"{count} faces (got {len(ret)})")

    return ret, pos


def _parse_binary_numeric_list(
    contents: bytes | bytearray,
    pos: int,
    *,
    dtype: type[_NumpyDType],
    elshape: _ElShape,
    empty_ok: bool = False,
) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[_NumpyDType]], int]:
    count, pos = _parse_number(contents, pos, target=int)
    if count < 0:
        raise ParseError(contents, pos, expected="non-negative list count")
    if count == 0 and not empty_ok:
        raise ParseError(contents, pos, expected="non-empty numeric list")
    pos = _skip(contents, pos)
    pos = _expect(contents, pos, b"(")

    if elshape:
        (dim,) = elshape
        elsize = dim
    else:
        elsize = 1
    items = count * elsize
    end = pos + items * np.dtype(dtype).itemsize

    _ = _expect(contents, end, b")")

    ret = np.frombuffer(contents, dtype=dtype, count=items, offset=pos).copy()

    if elshape:
        ret = ret.reshape((-1, *elshape))

    return ret, end + 1


def _parse_tensor(contents: bytes | bytearray, pos: int) -> tuple[Tensor, int]:
    if contents[pos : pos + 1] == b"(":
        pos += 1
        values: list[float] = []
        for _ in range(9):
            pos = _skip(contents, pos)
            try:
                value, pos = _parse_number(contents, pos, target=float)
            except ParseError:
                break
            values.append(value)

        pos = _skip(contents, pos)
        pos = _expect(contents, pos, b")")

        if len(values) not in (3, 6, 9):
            raise ParseError(contents, pos, expected="3, 6, or 9 values for tensor")

        return np.array(values), pos

    try:
        return _parse_number(contents, pos, target=float)
    except ParseError as e:
        raise ParseError(contents, pos, expected="tensor") from e


def _parse_field(contents: bytes | bytearray, pos: int) -> tuple[Field, int]:
    token, pos = _parse_token(contents, pos)
    match token:
        case "uniform":
            pos = _skip(contents, pos)
            return _parse_tensor(contents, pos)

        case "nonuniform":
            pos = _skip(contents, pos)
            token, pos = _parse_token(contents, pos)

            match token:
                case "List<scalar>":
                    elshape = ()
                    pos = _skip(contents, pos)
                    with contextlib.suppress(ParseError):
                        return _parse_ascii_float_list(contents, pos, empty_ok=True)
                case "List<vector>":
                    elshape = (3,)
                    pos = _skip(contents, pos)
                    with contextlib.suppress(ParseError):
                        return _parse_ascii_vector_list(contents, pos, empty_ok=True)
                case "List<symmTensor>":
                    elshape = (6,)
                    pos = _skip(contents, pos)
                    with contextlib.suppress(ParseError):
                        return _parse_ascii_symm_tensor_list(
                            contents, pos, empty_ok=True
                        )
                case "List<tensor>":
                    elshape = (9,)
                    pos = _skip(contents, pos)
                    with contextlib.suppress(ParseError):
                        return _parse_ascii_tensor_list(contents, pos, empty_ok=True)
                case _:
                    raise ParseError(
                        contents,
                        pos,
                        expected="one of: List<scalar>, List<vector>, List<symmTensor>, or List<tensor>",
                    )

            with contextlib.suppress(ParseError):
                return _parse_binary_numeric_list(
                    contents, pos, dtype=np.float64, elshape=elshape, empty_ok=True
                )
            return _parse_binary_numeric_list(
                contents, pos, dtype=np.float32, elshape=elshape, empty_ok=True
            )

        case _:
            raise ParseError(contents, pos, expected="'uniform' or 'nonuniform'")


def _parse_list(
    contents: bytes | bytearray, pos: int
) -> tuple[list[DataEntry | KeywordEntry | Dict], int]:
    try:
        count, pos = _parse_number(contents, pos, target=int)
    except ParseError:
        count = None
    else:
        if count < 0:
            raise ParseError(contents, pos, expected="non-negative list count")
        pos = _skip(contents, pos)

    if contents[pos : pos + 1] == b"(":
        pos += 1
        ret: list[DataEntry | KeywordEntry | Dict] = []
        while count is None or len(ret) < count:
            pos = _skip(contents, pos)
            if count is None and contents[pos : pos + 1] == b")":
                pos += 1
                break

            try:
                item, pos = _parse_dictionary(contents, pos)
            except ParseError:
                try:
                    item, pos = _parse_keyword_entry(contents, pos)
                except ParseError:
                    item, pos = _parse_data_entry(contents, pos)

            ret.append(item)

        if count is not None:
            pos = _skip(contents, pos)
            pos = _expect(contents, pos, b")")

    elif count is not None and contents[pos : pos + 1] == b"{":
        pos += 1
        pos = _skip(contents, pos)
        item: DataEntry | KeywordEntry | Dict
        try:
            item, pos = _parse_dictionary(contents, pos)
        except ParseError:
            try:
                item, pos = _parse_keyword_entry(contents, pos)
            except ParseError:
                item, pos = _parse_data_entry(contents, pos)
        pos = _skip(contents, pos)
        pos = _expect(contents, pos, b"}")

        ret: list[DataEntry | KeywordEntry | Dict] = [item]
        ret *= count

    else:
        raise ParseError(contents, pos, expected="list")

    return ret, pos


def _parse_dimensions(
    contents: bytes | bytearray, pos: int
) -> tuple[DimensionSet, int]:
    pos = _expect(contents, pos, b"[")

    dimensions = []
    for _ in range(7):
        pos = _skip(contents, pos)
        if contents[pos : pos + 1] == b"]":
            break

        try:
            dim, pos = _parse_number(contents, pos)
        except ParseError:
            break
        dimensions.append(dim)

    pos = _skip(contents, pos)
    pos = _expect(contents, pos, b"]")

    return DimensionSet(*dimensions), pos


def _parse_dimensioned(
    contents: bytes | bytearray, pos: int
) -> tuple[Dimensioned, int]:
    try:
        name, pos = _parse_token(contents, pos)
    except ParseError:
        name = None
    else:
        pos = _skip(contents, pos)
    dimensions, pos = _parse_dimensions(contents, pos)
    pos = _skip(contents, pos)
    value, pos = _parse_tensor(contents, pos)

    return Dimensioned(value, dimensions, name), pos


def _parse_switch(contents: bytes | bytearray, pos: int) -> tuple[bool, int]:
    token, pos = _parse_token(contents, pos)
    match token:
        case "yes" | "true" | "on":
            return True, pos
        case "no" | "false" | "off":
            return False, pos
        case _:
            raise ParseError(contents, pos, expected="switch value")


def _parse_keyword_entry(
    contents: bytes | bytearray, pos: int
) -> tuple[KeywordEntry, int]:
    keyword, pos = _parse_data_entry(contents, pos)
    pos = _skip(contents, pos)
    try:
        value, pos = _parse_dictionary(contents, pos)
    except ParseError:
        value, pos = _parse_data(contents, pos)
        pos = _skip(contents, pos)
        pos = _expect(contents, pos, b";")

    return (keyword, value), pos


def _parse_dictionary(contents: bytes | bytearray, pos: int) -> tuple[Dict, int]:
    pos = _expect(contents, pos, b"{")

    ret: Dict = {}
    while True:
        pos = _skip(contents, pos)
        if contents[pos : pos + 1] == b"}":
            pos += 1
            break

        keyword, pos = _parse_token(contents, pos)

        if keyword.startswith("#"):
            raise FoamFileDecodeError(
                contents,
                pos,
                expected=f"keyword not starting with # (got {keyword!r})",
            )

        if keyword in ret:
            raise FoamFileDecodeError(
                contents,
                pos,
                expected=f"non-duplicate keyword in dictionary (got {keyword!r})",
            )

        pos = _skip(contents, pos)

        try:
            value, pos = _parse_dictionary(contents, pos)
        except ParseError:
            value, pos = _parse_data(contents, pos)
            pos = _skip(contents, pos)
            pos = _expect(contents, pos, b";")

        ret[keyword] = value

    return ret, pos


def _parse_data_entry(contents: bytes | bytearray, pos: int) -> tuple[DataEntry, int]:
    for parser in (
        _parse_field,
        _parse_list,
        _parse_dimensioned,
        _parse_dimensions,
        _parse_number,
        _parse_switch,
    ):
        with contextlib.suppress(ParseError):
            return parser(contents, pos)

    return _parse_token(contents, pos)


def _parse_data(contents: bytes | bytearray, pos: int) -> tuple[Data, int]:
    entry, pos = _parse_data_entry(contents, pos)
    entries: list[DataEntry] = [entry]

    while True:
        pos = _skip(contents, pos)
        try:
            entry, pos = _parse_data_entry(contents, pos)
        except ParseError:
            break
        entries.append(entry)

    if len(entries) == 1:
        return entries[0], pos

    return tuple(entries), pos


def _parse_subdictionary(contents: bytes | bytearray, pos: int) -> tuple[SubDict, int]:
    pos = _expect(contents, pos, b"{")

    ret: SubDict = {}
    while True:
        pos = _skip(contents, pos)
        if contents[pos : pos + 1] == b"}":
            pos += 1
            break

        keyword, pos = _parse_token(contents, pos)

        if not keyword.startswith("#") and keyword in ret:
            raise FoamFileDecodeError(
                contents,
                pos,
                expected=f"non-duplicate keyword in subdictionary (got {keyword!r})",
            )

        pos = _skip(contents, pos)

        if keyword.startswith("#"):
            value, pos = _parse_data_entry(contents, pos)
            pos = _skip(contents, pos, newline_ok=False)
            if pos < len(contents):
                pos = _expect(contents, pos, b"\n")
        else:
            try:
                value, pos = _parse_subdictionary(contents, pos)
            except ParseError:
                try:
                    value, pos = _parse_data(contents, pos)
                except ParseError:
                    value = None
                else:
                    pos = _skip(contents, pos)
                pos = _expect(contents, pos, b";")

        ret = add_to_mapping(ret, keyword, value)  # ty: ignore[invalid-assignment]

    return ret, pos


def _parse_standalone_data_entry(
    contents: bytes | bytearray, pos: int
) -> tuple[StandaloneDataEntry, int]:
    with contextlib.suppress(ParseError):
        return _parse_ascii_integer_list(contents, pos)
    with contextlib.suppress(ParseError):
        return _parse_ascii_float_list(contents, pos)
    with contextlib.suppress(ParseError):
        return _parse_ascii_vector_list(contents, pos)
    with contextlib.suppress(ParseError):
        return _parse_ascii_faces_like_list(contents, pos)

    try:
        entry1, pos1 = _parse_data(contents, pos)
    except ParseError:
        pos1 = None

    try:
        entry2, pos2 = _parse_binary_numeric_list(
            contents, pos, dtype=np.int32, elshape=()
        )
    except ParseError:
        try:
            entry2, pos2 = _parse_binary_numeric_list(
                contents, pos, dtype=np.float64, elshape=()
            )
        except ParseError:
            try:
                entry2, pos2 = _parse_binary_numeric_list(
                    contents, pos, dtype=np.float64, elshape=(3,)
                )
            except ParseError:
                pos2 = None

    match pos1, pos2:
        case None, None:
            raise ParseError(contents, pos, expected="standalone data entry")
        case _, None:
            return entry1, pos1
        case None, _:
            return entry2, pos2
        case _ if pos1 > pos2:
            return entry1, pos1
        case _:
            return entry2, pos2


def _parse_standalone_data(
    contents: bytes | bytearray, pos: int
) -> tuple[StandaloneData, int]:
    entry, pos = _parse_standalone_data_entry(contents, pos)
    entries: list[StandaloneDataEntry] = [entry]

    while True:
        pos = _skip(contents, pos)
        try:
            entry, pos = _parse_standalone_data_entry(contents, pos)
        except ParseError:
            break
        entries.append(entry)

    if len(entries) == 1:
        return entries[0], pos

    return tuple(entries), pos


def _parse_file(contents: bytes | bytearray, pos: int = 0) -> tuple[FileDict, int]:
    ret: FileDict = {}

    while (pos := _skip(contents, pos)) < len(contents):
        try:
            keyword, new_pos = _parse_token(contents, pos)
            if not keyword.startswith("#") and keyword in ret:
                raise FoamFileDecodeError(
                    contents,
                    pos,
                    expected=f"non-duplicate keyword in file (got {keyword!r})",
                )

            new_pos = _skip(contents, new_pos)

            if keyword.startswith("#"):
                value, new_pos = _parse_data_entry(contents, new_pos)
                new_pos = _skip(contents, new_pos, newline_ok=False)
                if new_pos < len(contents):
                    new_pos = _expect(contents, new_pos, b"\n")
            else:
                try:
                    value, new_pos = _parse_subdictionary(contents, new_pos)
                except ParseError:
                    try:
                        value, new_pos = _parse_data(contents, new_pos)
                    except ParseError:
                        value = None
                    else:
                        new_pos = _skip(contents, new_pos)
                    new_pos = _expect(contents, new_pos, b";")
            ret = add_to_mapping(ret, keyword, value)  # ty: ignore[invalid-assignment]
            pos = new_pos
        except ParseError:  # noqa: PERF203
            try:
                standalone_data, pos = _parse_standalone_data(contents, pos)
            except ParseError:
                raise ParseError(
                    contents, pos, expected="keyword or standalone data"
                ) from None
            else:
                if None in ret:
                    raise FoamFileDecodeError(
                        contents,
                        pos,
                        expected="only one standalone data block",
                    )
                ret[None] = standalone_data

    return ret, pos


def parse(contents: bytes | bytearray | str, /, *, target: type[_Output]) -> _Output:
    if isinstance(contents, str):
        contents = contents.encode()

    try:
        pos = _skip(contents, 0)
        if target == FileDict:
            ret, pos = _parse_file(contents, pos)
        elif target == Data:
            ret, pos = _parse_data(contents, pos)
        elif target == StandaloneData:
            ret, pos = _parse_standalone_data(contents, pos)
        elif target is str:
            ret, pos = _parse_token(contents, pos)
        else:
            assert_never(target)  # ty: ignore[type-assertion-failure]
        _skip(contents, pos)
    except ParseError as e:
        raise e.make_fatal() from None

    if pos != len(contents):
        raise FoamFileDecodeError(
            contents,
            pos,
            expected="end of file",
        )

    return ret  # ty: ignore[invalid-return-type]


@dataclasses.dataclass
class ParsedEntry:
    """Represents a parsed entry with data and location information."""

    data: Data | StandaloneData | EllipsisType | None
    start: int
    end: int


def _parse_file_located(
    contents: bytes | bytearray,
    pos: int,
    _keywords: tuple[str, ...] = (),
) -> tuple[MultiDict[tuple[str, ...], ParsedEntry], int]:
    ret: MultiDict[tuple[str, ...], ParsedEntry] = MultiDict()

    while (pos := _skip(contents, pos)) < len(contents):
        # Check if we've hit a closing brace (end of subdictionary)
        if _keywords and contents[pos : pos + 1] == b"}":
            return ret, pos

        entry_start = pos
        try:
            keyword, new_pos = _parse_token(contents, pos)
            new_pos = _skip(contents, new_pos)

            if keyword.startswith("#"):
                value, new_pos = _parse_data_entry(contents, new_pos)
                new_pos = _skip(contents, new_pos, newline_ok=False)
                # Expect newline or end for directives
                if new_pos < len(contents):
                    new_pos = _expect(contents, new_pos, b"\n")
                ret.add((*_keywords, keyword), ParsedEntry(value, entry_start, new_pos))
            # Check if this is a subdictionary
            elif contents[new_pos : new_pos + 1] == b"{":
                # Check for duplicates
                if (*_keywords, keyword) in ret:
                    raise FoamFileDecodeError(
                        contents,
                        entry_start,
                        expected=f"non-duplicate entry for keyword: {keyword}",
                    )

                # Skip opening brace
                new_pos += 1

                # Recursively parse subdictionary content
                # The recursive call will parse until it hits the closing brace
                subdict_result, new_pos = _parse_file_located(
                    contents,
                    new_pos,
                    (*_keywords, keyword),
                )

                # Expect closing brace
                new_pos = _skip(contents, new_pos)
                new_pos = _expect(contents, new_pos, b"}")

                # Add entry with ... marker for subdictionary
                ret[(*_keywords, keyword)] = ParsedEntry(..., entry_start, new_pos)
                ret.extend(subdict_result)
            else:
                try:
                    value, new_pos = _parse_data(contents, new_pos)
                except ParseError:
                    value = None
                else:
                    new_pos = _skip(contents, new_pos)
                new_pos = _expect(contents, new_pos, b";")

                # Check for duplicates
                if (*_keywords, keyword) in ret and not keyword.startswith("#"):
                    raise FoamFileDecodeError(
                        contents,
                        entry_start,
                        expected=f"non-duplicate entry for keyword: {keyword}",
                    )

                ret.add((*_keywords, keyword), ParsedEntry(value, entry_start, new_pos))

            pos = new_pos
        except ParseError:
            # If keyword parsing fails, try parsing as standalone data
            # This pattern is necessary because OpenFOAM files can contain
            # standalone data (numeric arrays, etc.) without keywords
            if _keywords:
                # Inside subdictionary - can't parse keyword, likely at closing brace or invalid syntax
                # Let it fail rather than silently skipping content
                break

            try:
                standalone_data, new_pos = _parse_standalone_data(contents, pos)
            except ParseError:
                raise ParseError(
                    contents, pos, expected="keyword or standalone data"
                ) from None
            else:
                if () in ret:
                    raise FoamFileDecodeError(
                        contents,
                        pos,
                        expected="only one standalone data block",
                    )
                ret[()] = ParsedEntry(standalone_data, entry_start, new_pos)
                pos = new_pos

    return ret, pos


def parse_located(
    contents: bytes | bytearray, /
) -> MultiDict[tuple[str, ...], ParsedEntry]:
    try:
        pos = _skip(contents, 0)
        ret, pos = _parse_file_located(contents, pos)
        _skip(contents, pos)
    except ParseError as e:
        raise e.make_fatal() from None
    if pos != len(contents):
        raise FoamFileDecodeError(
            contents,
            pos,
            expected="end of file",
        )
    return ret
