import builtins
import contextlib
import dataclasses
import sys
from typing import Literal, TypeVar, overload

from foamlib._files._util import add_to_mapping

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

import numpy as np

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
from ._exceptions import ParseError
from ._re import (
    COMMENT,
    FACES_LIKE_LIST,
    FLOAT,
    FLOAT_LIST,
    INTEGER,
    INTEGER_LIST,
    SKIP,
    SKIP_NO_NEWLINE,
    SYMM_TENSOR_LIST,
    TENSOR_LIST,
    TOKEN,
    UNCOMMENTED_FACES_LIKE_LIST,
    UNCOMMENTED_FLOAT_LIST,
    UNCOMMENTED_INTEGER_LIST,
    UNCOMMENTED_SYMM_TENSOR_LIST,
    UNCOMMENTED_TENSOR_LIST,
    UNCOMMENTED_VECTOR_LIST,
    UNSIGNED_INTEGER,
    VECTOR_LIST,
)

_DT = TypeVar("_DT", np.float64, np.float32, np.int64, np.int32)
_ElShape = TypeVar(
    "_ElShape", tuple[()], tuple[Literal[3]], tuple[Literal[6]], tuple[Literal[9]]
)


def skip(
    contents: bytes, pos: int, *, strict: bool = False, newline_ok: bool = True
) -> int:
    pattern = SKIP if newline_ok else SKIP_NO_NEWLINE

    if match := pattern.match(contents, pos):
        return match.end()

    if strict and pos < len(contents):
        raise ParseError(contents, pos, expected="whitespace or comment")

    return pos


def _expect(contents: bytes, pos: int, expected: bytes) -> int:
    length = len(expected)
    if pos >= len(contents) or contents[pos : pos + length] != expected:
        raise ParseError(contents, pos, expected=repr(expected.decode("ascii")))

    return pos + length


@overload
def _parse_ascii_numeric_list(
    contents: bytes,
    pos: int,
    *,
    dtype: type[float],
    elshape: _ElShape,
    empty_ok: bool = False,
) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.float64]], int]: ...


@overload
def _parse_ascii_numeric_list(
    contents: bytes,
    pos: int,
    *,
    dtype: type[int],
    elshape: _ElShape,
    empty_ok: bool = False,
) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.int64]], int]: ...


def _parse_ascii_numeric_list(
    contents: bytes,
    pos: int,
    *,
    dtype: type[float] | type[int],
    elshape: _ElShape,
    empty_ok: bool = False,
) -> tuple[
    np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[np.float64 | np.int64]], int
]:
    try:
        count, pos = _parse_unsigned_integer(contents, pos)
    except ParseError:
        count = None
    else:
        if count == 0 and not empty_ok:
            raise ParseError(contents, pos, expected="non-empty numeric list")
        pos = skip(contents, pos)

    if contents[pos : pos + 1] == b"(":
        match dtype, elshape:
            case builtins.float, ():
                pattern = FLOAT_LIST
                uncommented_pattern = UNCOMMENTED_FLOAT_LIST
            case builtins.float, (3,):
                pattern = VECTOR_LIST
                uncommented_pattern = UNCOMMENTED_VECTOR_LIST
            case builtins.float, (6,):
                pattern = SYMM_TENSOR_LIST
                uncommented_pattern = UNCOMMENTED_SYMM_TENSOR_LIST
            case builtins.float, (9,):
                pattern = TENSOR_LIST
                uncommented_pattern = UNCOMMENTED_TENSOR_LIST
            case builtins.int, ():
                pattern = INTEGER_LIST
                uncommented_pattern = UNCOMMENTED_INTEGER_LIST
            case _:
                raise NotImplementedError

        if match := uncommented_pattern.match(contents, pos):
            data = contents[pos + 1 : match.end() - 1]
            pos = match.end()

        elif match := pattern.match(contents, pos):
            data = contents[pos + 1 : match.end() - 1]
            pos = match.end()

            data = COMMENT.sub(b" ", data)

        if not match:
            raise ParseError(
                contents,
                pos,
                expected=f"numeric list of type {dtype} and shape {elshape}",
            )

        if elshape:
            data = data.replace(b"(", b" ").replace(b")", b" ")

        ret = np.fromstring(data, sep=" ", dtype=dtype)

        if elshape:
            ret = ret.reshape((-1, *elshape))

        if not empty_ok and len(ret) == 0:
            raise ParseError(contents, pos, expected="non-empty numeric list")

        if count is not None and len(ret) != count:
            raise ParseError(
                contents, pos, expected=f"{count} elements (got {len(ret)})"
            )

    elif count is not None and contents[pos : pos + 1] == b"{":
        pos += 1
        pos = skip(contents, pos)

        match dtype:
            case builtins.float:
                parse_number = _parse_float
            case builtins.int:
                parse_number = _parse_integer
            case _:
                raise NotImplementedError

        if elshape:
            elem = []
            pos = _expect(contents, pos, b"(")
            for _ in range(elshape[0]):
                pos = skip(contents, pos)
                x, pos = parse_number(contents, pos)
                elem.append(x)
            pos = skip(contents, pos)
            pos = _expect(contents, pos, b")")
        else:
            elem, pos = parse_number(contents, pos)

        pos = _expect(contents, pos, b"}")

        ret = np.full((count, *elshape), elem, dtype=dtype)

    else:
        raise ParseError(contents, pos, expected="ASCII numeric list")

    return ret, pos


def _parse_ascii_faces_like_list(
    contents: bytes, pos: int
) -> tuple[list[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]]], int]:
    try:
        count, pos = _parse_unsigned_integer(contents, pos)
    except ParseError:
        count = None
    else:
        pos = skip(contents, pos)

    _ = _expect(contents, pos, b"(")

    if match := UNCOMMENTED_FACES_LIKE_LIST.match(contents, pos):
        data = contents[pos + 1 : match.end() - 1]
        pos = match.end()

    elif match := FACES_LIKE_LIST.match(contents, pos):
        data = contents[pos + 1 : match.end() - 1]
        pos = match.end()

        data = COMMENT.sub(b" ", data)

    if not match:
        raise ParseError(contents, pos, expected="faces-like list")

    data = data.replace(b"(", b" ").replace(b")", b" ")
    values = np.fromstring(data, sep=" ", dtype=int)

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
    contents: bytes,
    pos: int,
    *,
    dtype: type[_DT],
    elshape: _ElShape,
    empty_ok: bool = False,
) -> tuple[np.ndarray[tuple[int, Unpack[_ElShape]], np.dtype[_DT]], int]:
    count, pos = _parse_unsigned_integer(contents, pos)
    if count == 0 and not empty_ok:
        raise ParseError(contents, pos, expected="non-empty numeric list")
    pos = skip(contents, pos)
    pos = _expect(contents, pos, b"(")

    elsize = np.dtype(dtype).itemsize
    if elshape:
        (dim,) = elshape
        elsize *= dim
    byte_count = count * elsize
    _ = _expect(contents, pos + byte_count, b")")

    ret = np.frombuffer(contents[pos : pos + byte_count], dtype=dtype)
    if elshape:
        ret = ret.reshape((-1, *elshape))

    pos += byte_count + 1
    return ret, pos


def _parse_tensor(contents: bytes, pos: int) -> tuple[Tensor, int]:
    if contents[pos : pos + 1] == b"(":
        pos += 1
        values: list[float] = []
        for _ in range(9):
            pos = skip(contents, pos)
            try:
                value, pos = _parse_float(contents, pos)
            except ParseError:
                break
            values.append(value)

        pos = skip(contents, pos)
        pos = _expect(contents, pos, b")")

        if len(values) not in (3, 6, 9):
            raise ParseError(contents, pos, expected="3, 6, or 9 values for tensor")

        return np.array(values), pos

    try:
        return _parse_float(contents, pos)
    except ParseError as e:
        raise ParseError(contents, pos, expected="tensor") from e


def _parse_field(contents: bytes, pos: int) -> tuple[Field, int]:
    token, pos = parse_token(contents, pos)
    match token:
        case "uniform":
            pos = skip(contents, pos)
            return _parse_tensor(contents, pos)

        case "nonuniform":
            pos = skip(contents, pos)
            token, pos = parse_token(contents, pos)

            match token:
                case "List<scalar>":
                    elshape = ()
                case "List<vector>":
                    elshape = (3,)
                case "List<symmTensor>":
                    elshape = (6,)
                case "List<tensor>":
                    elshape = (9,)
                case _:
                    raise ParseError(
                        contents,
                        pos,
                        expected="one of: List<scalar>, List<vector>, List<symmTensor>, or List<tensor>",
                    )

            pos = skip(contents, pos)
            with contextlib.suppress(ParseError):
                return _parse_ascii_numeric_list(
                    contents, pos, dtype=float, elshape=elshape, empty_ok=True
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


def parse_token(contents: bytes, pos: int) -> tuple[str, int]:
    if match := TOKEN.match(contents, pos):
        return match.group(0).decode("ascii"), match.end()

    raise ParseError(contents, pos, expected="token")


def _parse_unsigned_integer(contents: bytes, pos: int) -> tuple[int, int]:
    if match := UNSIGNED_INTEGER.match(contents, pos):
        return int(match.group(0).decode("ascii")), match.end()

    raise ParseError(contents, pos, expected="unsigned integer")


def _parse_integer(contents: bytes, pos: int) -> tuple[int, int]:
    if match := INTEGER.match(contents, pos):
        return int(match.group(0).decode("ascii")), match.end()

    raise ParseError(contents, pos, expected="integer")


def _parse_float(contents: bytes, pos: int) -> tuple[float, int]:
    if match := FLOAT.match(contents, pos):
        return float(match.group(0).decode("ascii")), match.end()

    raise ParseError(contents, pos, expected="float")


def _parse_list(
    contents: bytes, pos: int
) -> tuple[list[DataEntry | KeywordEntry | Dict], int]:
    try:
        count, pos = _parse_unsigned_integer(contents, pos)
    except ParseError:
        count = None
    else:
        pos = skip(contents, pos)

    if contents[pos : pos + 1] == b"(":
        pos += 1
        ret: list[DataEntry | KeywordEntry | Dict] = []
        while True:
            pos = skip(contents, pos)
            if contents[pos : pos + 1] == b")":
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

    elif count is not None and contents[pos : pos + 1] == b"{":
        pos += 1
        pos = skip(contents, pos)
        item: DataEntry | KeywordEntry | Dict
        try:
            item, pos = _parse_dictionary(contents, pos)
        except ParseError:
            try:
                item, pos = _parse_keyword_entry(contents, pos)
            except ParseError:
                item, pos = _parse_data_entry(contents, pos)
        pos = skip(contents, pos)
        pos = _expect(contents, pos, b"}")

        ret: list[DataEntry | KeywordEntry | Dict] = [item]
        ret *= count

    else:
        raise ParseError(contents, pos, expected="list")

    return ret, pos


def _parse_dimensions(contents: bytes, pos: int) -> tuple[DimensionSet, int]:
    pos = _expect(contents, pos, b"[")

    dimensions = []
    for _ in range(7):
        pos = skip(contents, pos)
        if contents[pos : pos + 1] == b"]":
            break

        try:
            dim, pos = _parse_integer(contents, pos)
        except ParseError:
            try:
                dim, pos = _parse_float(contents, pos)
            except ParseError:
                break
        dimensions.append(dim)

    pos = skip(contents, pos)
    pos = _expect(contents, pos, b"]")

    return DimensionSet(*dimensions), pos


def _parse_dimensioned(contents: bytes, pos: int) -> tuple[Dimensioned, int]:
    try:
        name, pos = parse_token(contents, pos)
    except ParseError:
        name = None
    else:
        pos = skip(contents, pos)
    dimensions, pos = _parse_dimensions(contents, pos)
    pos = skip(contents, pos)
    value, pos = _parse_tensor(contents, pos)

    return Dimensioned(value, dimensions, name), pos


def _parse_switch(contents: bytes, pos: int) -> tuple[bool, int]:
    token, pos = parse_token(contents, pos)
    match token:
        case "yes" | "true" | "on":
            return True, pos
        case "no" | "false" | "off":
            return False, pos
        case _:
            raise ParseError(contents, pos, expected="switch value")


def _parse_keyword_entry(contents: bytes, pos: int) -> tuple[KeywordEntry, int]:
    keyword, pos = _parse_data_entry(contents, pos)
    pos = skip(contents, pos)
    try:
        value, pos = _parse_dictionary(contents, pos)
    except ParseError:
        value, pos = parse_data(contents, pos)
        pos = skip(contents, pos)
        pos = _expect(contents, pos, b";")

    return (keyword, value), pos


def _parse_dictionary(contents: bytes, pos: int) -> tuple[Dict, int]:
    pos = _expect(contents, pos, b"{")

    ret: Dict = {}
    while True:
        pos = skip(contents, pos)
        if contents[pos : pos + 1] == b"}":
            pos += 1
            break

        keyword, pos = parse_token(contents, pos)

        if keyword.startswith("#"):
            raise ParseError(
                contents, pos, expected="non-directive keyword in dictionary"
            )

        if keyword in ret:
            raise ParseError(
                contents, pos, expected=f"unique keyword (got duplicate {keyword!r})"
            )

        pos = skip(contents, pos)

        try:
            value, pos = _parse_dictionary(contents, pos)
        except ParseError:
            value, pos = parse_data(contents, pos)
            pos = skip(contents, pos)
            pos = _expect(contents, pos, b";")

        ret[keyword] = value

    return ret, pos


def _parse_data_entry(contents: bytes, pos: int) -> tuple[DataEntry, int]:
    for parser in (
        _parse_field,
        _parse_list,
        _parse_dimensioned,
        _parse_dimensions,
        _parse_integer,
        _parse_float,
        _parse_switch,
    ):
        with contextlib.suppress(ParseError):
            return parser(contents, pos)

    return parse_token(contents, pos)


def parse_data(contents: bytes, pos: int) -> tuple[Data, int]:
    entry, pos = _parse_data_entry(contents, pos)
    entries: list[DataEntry] = [entry]

    while True:
        pos = skip(contents, pos)
        try:
            entry, pos = _parse_data_entry(contents, pos)
        except ParseError:
            break
        entries.append(entry)

    if len(entries) == 1:
        return entries[0], pos

    return tuple(entries), pos


def _parse_subdictionary(contents: bytes, pos: int) -> tuple[SubDict, int]:
    pos = _expect(contents, pos, b"{")

    ret: SubDict = {}
    while True:
        pos = skip(contents, pos)
        if contents[pos : pos + 1] == b"}":
            pos += 1
            break

        keyword, pos = parse_token(contents, pos)

        if not keyword.startswith("#") and keyword in ret:
            raise ParseError(
                contents, pos, expected=f"unique keyword (got duplicate {keyword!r})"
            )

        pos = skip(contents, pos)

        if keyword.startswith("#"):
            value, pos = _parse_data_entry(contents, pos)
            pos = skip(contents, pos, newline_ok=False)
            pos = _expect(contents, pos, b"\n")
        else:
            try:
                value, pos = _parse_subdictionary(contents, pos)
            except ParseError:
                try:
                    value, pos = parse_data(contents, pos)
                except ParseError:
                    value = None
                else:
                    pos = skip(contents, pos)
                pos = _expect(contents, pos, b";")

        ret = add_to_mapping(ret, keyword, value)  # ty: ignore[invalid-assignment]

    return ret, pos


def _parse_standalone_data_entry(
    contents: bytes, pos: int
) -> tuple[StandaloneDataEntry, int]:
    with contextlib.suppress(ParseError):
        return _parse_ascii_numeric_list(contents, pos, dtype=int, elshape=())
    with contextlib.suppress(ParseError):
        return _parse_ascii_numeric_list(contents, pos, dtype=float, elshape=())
    with contextlib.suppress(ParseError):
        return _parse_ascii_numeric_list(contents, pos, dtype=float, elshape=(3,))
    with contextlib.suppress(ParseError):
        return _parse_ascii_faces_like_list(contents, pos)

    try:
        entry1, pos1 = parse_data(contents, pos)
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


def parse_standalone_data(contents: bytes, pos: int) -> tuple[StandaloneData, int]:
    entry, pos = _parse_standalone_data_entry(contents, pos)
    entries: list[StandaloneDataEntry] = [entry]

    while True:
        pos = skip(contents, pos)
        try:
            entry, pos = _parse_standalone_data_entry(contents, pos)
        except ParseError:
            break
        entries.append(entry)

    if len(entries) == 1:
        return entries[0], pos

    return tuple(entries), pos


def parse_file(contents: bytes, pos: int = 0) -> tuple[FileDict, int]:
    ret: FileDict = {}

    while (pos := skip(contents, pos)) < len(contents):
        try:
            keyword, new_pos = parse_token(contents, pos)
            if not keyword.startswith("#") and keyword in ret:
                raise ParseError(  # noqa: TRY301
                    contents,
                    pos,
                    expected=f"unique non-directive keyword (got duplicate {keyword!r})",
                )

            new_pos = skip(contents, new_pos)

            if keyword.startswith("#"):
                value, new_pos = _parse_data_entry(contents, new_pos)
                new_pos = skip(contents, new_pos, newline_ok=False)
                new_pos = _expect(contents, new_pos, b"\n")
            else:
                try:
                    value, new_pos = _parse_subdictionary(contents, new_pos)
                except ParseError:
                    try:
                        value, new_pos = parse_data(contents, new_pos)
                    except ParseError:
                        value = None
                    else:
                        new_pos = skip(contents, new_pos)
                    new_pos = _expect(contents, new_pos, b";")
            ret = add_to_mapping(ret, keyword, value)  # ty: ignore[invalid-assignment]
            pos = new_pos
        except ParseError:  # noqa: PERF203
            try:
                standalone_data, pos = parse_standalone_data(contents, pos)
            except ParseError:
                raise ParseError(
                    contents, pos, expected="keyword or standalone data"
                ) from None
            else:
                if None in ret:
                    raise ParseError(
                        contents,
                        pos,
                        expected="non-duplicate top-level standalone data",
                    ) from None
                ret[None] = standalone_data

    return ret, pos


@dataclasses.dataclass
class LocatedEntry:
    """Represents a parsed entry with its location in the source."""

    value: tuple[str | None, Data | StandaloneData | Dict | SubDict | None]
    locn_start: int
    locn_end: int


def _find_matching_brace(contents: bytes, pos: int) -> int:
    """Find the position of the matching closing brace for an opening brace.

    Args:
        contents: The bytes to search in
        pos: Position of the opening brace

    Returns:
        Position right after the closing brace

    Raises:
        ParseError: If no matching brace is found
    """
    assert contents[pos : pos + 1] == b"{"
    depth = 1
    pos += 1

    while pos < len(contents) and depth > 0:
        # Skip comments
        if contents[pos : pos + 2] == b"/*":
            end = contents.find(b"*/", pos + 2)
            if end == -1:
                raise ParseError(contents, pos, expected="closing */")
            pos = end + 2
            continue
        if contents[pos : pos + 2] == b"//":
            end = contents.find(b"\n", pos + 2)
            pos = len(contents) if end == -1 else end + 1
            continue

        if contents[pos : pos + 1] == b"{":
            depth += 1
        elif contents[pos : pos + 1] == b"}":
            depth -= 1

        pos += 1

    if depth != 0:
        raise ParseError(contents, pos, expected="matching closing brace")

    return pos


def parse_file_located(contents: bytes, pos: int = 0) -> list[LocatedEntry]:
    """Parse a file and return entries with location information.

    This function is similar to parse_file but returns a list of LocatedEntry
    objects that include start and end positions for each top-level entry.
    This is used by ParsedFile to track locations for editing.

    Args:
        contents: The bytes to parse
        pos: Starting position (default 0)

    Returns:
        A list of LocatedEntry objects containing parsed entries with locations
    """
    ret: list[LocatedEntry] = []

    while (pos := skip(contents, pos)) < len(contents):
        entry_start = pos
        try:
            keyword, new_pos = parse_token(contents, pos)
            new_pos = skip(contents, new_pos)

            if keyword.startswith("#"):
                value, new_pos = _parse_data_entry(contents, new_pos)
                new_pos = skip(contents, new_pos, newline_ok=False)
                # Expect newline or end of file for directives
                if new_pos < len(contents):
                    new_pos = _expect(contents, new_pos, b"\n")
            # Check if this is a subdictionary
            elif contents[new_pos : new_pos + 1] == b"{":
                # Just find the matching brace without parsing/validating the content
                # The content will be parsed later by _flatten_results
                new_pos = _find_matching_brace(contents, new_pos)
                value = {}  # Marker for subdictionary
            else:
                try:
                    value, new_pos = parse_data(contents, new_pos)
                except ParseError:
                    value = None
                else:
                    new_pos = skip(contents, new_pos)
                new_pos = _expect(contents, new_pos, b";")

            ret.append(LocatedEntry((keyword, value), entry_start, new_pos))
            pos = new_pos
        except ParseError:
            # If keyword parsing fails, try parsing as standalone data
            # This pattern is necessary because OpenFOAM files can contain
            # standalone data (numeric arrays, etc.) without keywords
            try:
                standalone_data, new_pos = parse_standalone_data(contents, pos)
            except ParseError:
                raise ParseError(
                    contents, pos, expected="keyword or standalone data"
                ) from None
            else:
                ret.append(LocatedEntry((None, standalone_data), entry_start, new_pos))
                pos = new_pos

    return ret
