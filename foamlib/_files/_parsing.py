from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Tuple, Union, cast, overload

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(...)

import numpy as np
from pyparsing import (
    CaselessKeyword,
    CharsNotIn,
    Combine,
    Dict,
    Forward,
    Group,
    Keyword,
    LineEnd,
    Literal,
    Located,
    NoMatch,
    Opt,
    ParserElement,
    ParseResults,
    Regex,
    Word,
    common,
    counted_array,
    dbl_quoted_string,
    identchars,
    printables,
)

from ._types import Data, Dimensioned, DimensionSet, File, StandaloneData, SubDict

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _ascii_numeric_list(
    dtype: DTypeLike,
    *,
    nested: int | None = None,
    ignore: Regex | None = None,
    empty_ok: bool = False,
) -> ParserElement:
    dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.floating):
        element = common.ieee_float
        element_pattern = r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))"
    elif np.issubdtype(dtype, np.integer):
        element = common.integer
        element_pattern = r"(?:-?\d+)"
    else:
        msg = f"Unsupported dtype: {dtype}"
        raise TypeError(msg)

    spacing_pattern = (
        rf"(?:(?:\s|{ignore.re.pattern})+)" if ignore is not None else r"(?:\s+)"
    )

    if nested is not None:
        element = (
            Literal("(").suppress() + Group(element[nested]) + Literal(")").suppress()
        )
        element_pattern = rf"(?:{spacing_pattern}?\({element_pattern}?(?:{element_pattern}{spacing_pattern}){{{nested - 1}}}{element_pattern}{spacing_pattern}?\))"

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_

        if not tks:
            count = None
        else:
            (count,) = tks
            assert isinstance(count, int)

        if count is None:
            if not empty_ok:
                list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}{spacing_pattern}?\)"
            else:
                list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}?{spacing_pattern}?\)"

        elif count == 0:
            if not empty_ok:
                list_ <<= NoMatch()
            else:
                list_ <<= (Literal("(") + Literal(")")).add_parse_action(
                    lambda: np.empty((0, nested) if nested else 0, dtype=dtype)
                )
            return

        else:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern}){{{count - 1}}}{element_pattern}{spacing_pattern}?\)"

        list_ <<= Regex(list_pattern).add_parse_action(
            lambda tks: to_array(tks, dtype=dtype)
        )

    def to_array(
        tks: ParseResults, *, dtype: DTypeLike
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        (s,) = tks
        assert s.startswith("(")
        assert s.endswith(")")
        s = s[1:-1]
        if ignore is not None:
            s = re.sub(ignore.re, " ", s)
        if nested is not None:
            s = s.replace("(", " ").replace(")", " ")

        ret: np.ndarray[
            tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]
        ] = np.fromstring(s, sep=" ", dtype=dtype)

        if nested is not None:
            ret = ret.reshape(-1, nested)

        return ret

    def to_full_array(
        tks: ParseResults, *, dtype: type
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        count, lst = tks
        assert isinstance(count, int)

        if nested is None:
            return np.full(count, lst, dtype=dtype)

        return np.full((count, nested), lst, dtype=dtype)  # type: ignore[return-value]

    ret = ((Opt(common.integer).add_parse_action(process_count)).suppress() + list_) | (
        common.integer + Literal("{").suppress() + element + Literal("}").suppress()
    ).add_parse_action(lambda tks: to_full_array(tks, dtype=float))

    if ignore is not None:
        ret.ignore(ignore)

    return ret


def _binary_numeric_list(
    dtype: DTypeLike, *, nested: int | None = None, empty_ok: bool = False
) -> ParserElement:
    dtype = np.dtype(dtype)

    elsize = nested if nested is not None else 1

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_
        (size,) = tks
        assert isinstance(size, int)

        if size == 0 and not empty_ok:
            list_ <<= NoMatch()
            return

        list_ <<= Regex(rf"\((?s:{'.' * dtype.itemsize * elsize}){{{size}}}\)")

    def to_array(
        tks: ParseResults,
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        size, s = tks
        assert isinstance(size, int)
        assert isinstance(s, str)
        assert s[0] == "("
        assert s[-1] == ")"
        s = s[1:-1]

        ret = np.frombuffer(s.encode("latin-1"), dtype=dtype)

        if nested is not None:
            ret = ret.reshape(-1, nested)

        return ret

    return (
        common.integer.copy().add_parse_action(process_count) + list_
    ).add_parse_action(to_array)


def _ascii_face_list(*, ignore: Regex | None = None) -> ParserElement:
    element_pattern = r"(?:-?\d+)"
    spacing_pattern = (
        rf"(?:(?:\s|{ignore.re.pattern})+)" if ignore is not None else r"(?:\s+)"
    )

    element_pattern = rf"(?:(?:3{spacing_pattern}?\((?:{element_pattern}{spacing_pattern}){{2}}{element_pattern}{spacing_pattern}?\))|(?:4{spacing_pattern}?\((?:{element_pattern}{spacing_pattern}){{3}}{element_pattern}{spacing_pattern}?\)))"

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_
        if not tks:
            count = None
        else:
            (count,) = tks
            assert isinstance(count, int)

        if count is None:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}{spacing_pattern}?\)"

        elif count == 0:
            list_ <<= NoMatch()
            return

        else:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern}){{{count - 1}}}{element_pattern}{spacing_pattern}?\)"

        list_ <<= Regex(list_pattern).add_parse_action(to_face_list)

    def to_face_list(
        tks: ParseResults,
    ) -> list[list[np.ndarray[tuple[int], np.dtype[np.int64]]]]:
        (s,) = tks
        assert s.startswith("(")
        assert s.endswith(")")
        if ignore is not None:
            s = re.sub(ignore.re, " ", s)
        s = s.replace("(", " ").replace(")", " ")

        raw = np.fromstring(s, sep=" ", dtype=int)

        values: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
        i = 0
        while i < raw.size:
            assert raw[i] in (3, 4)
            values.append(raw[i + 1 : i + raw[i] + 1])
            i += raw[i] + 1

        return [values]

    return Opt(common.integer).add_parse_action(process_count).suppress() + list_


def _list_of(entry: ParserElement) -> ParserElement:
    return (
        (
            counted_array(entry, common.integer + Literal("(").suppress())
            + Literal(")").suppress()
        ).set_parse_action(lambda tks: [tks.as_list()])
        | (
            Literal("(").suppress()
            + Group((entry)[...], aslist=True)
            + Literal(")").suppress()
        )
        | (
            common.integer + Literal("{").suppress() + entry + Literal("}").suppress()
        ).set_parse_action(lambda tks: [[tks[1]] * tks[0]])
    )


def _dict_of(
    keyword: ParserElement,
    data: ParserElement,
    *,
    directive: ParserElement | None = None,
    data_entry: ParserElement | None = None,
    located: bool = False,
) -> ParserElement:
    dict_ = Forward()

    keyword_entry = keyword + (dict_ | (data + Literal(";").suppress()))

    if directive is not None:
        assert data_entry is not None
        keyword_entry |= directive + data_entry + LineEnd().suppress()

    if located:
        keyword_entry = Located(keyword_entry)

    dict_ <<= (
        Literal("{").suppress()
        + Dict(Group(keyword_entry)[...], asdict=not located)
        + Literal("}").suppress()
    )

    return dict_


def _keyword_entry_of(
    keyword: ParserElement,
    data: ParserElement,
    *,
    directive: ParserElement | None = None,
    data_entry: ParserElement | None = None,
    located: bool = False,
) -> ParserElement:
    keyword_entry = keyword + (
        _dict_of(
            keyword, data, directive=directive, data_entry=data_entry, located=located
        )
        | (data + Literal(";").suppress())
    )

    if directive is not None:
        assert data_entry is not None
        keyword_entry |= directive + data_entry + LineEnd().suppress()

    if located:
        keyword_entry = Located(keyword_entry)
    else:
        keyword_entry = keyword_entry.copy().set_parse_action(lambda tks: tuple(tks))

    return keyword_entry


# https://github.com/pyparsing/pyparsing/pull/584
_COMMENT = Regex(r"(?:/\*(?:[^*]|\*(?!/))*\*/)|(?://(?:\\\n|[^\n])*)")

_IDENTCHARS = identchars + "$"
_IDENTBODYCHARS = (
    printables.replace(";", "")
    .replace("(", "")
    .replace(")", "")
    .replace("{", "")
    .replace("}", "")
    .replace("[", "")
    .replace("]", "")
)

_SWITCH = (
    Keyword("yes", _IDENTBODYCHARS)
    | Keyword("true", _IDENTBODYCHARS)
    | Keyword("on", _IDENTBODYCHARS)
).set_parse_action(lambda: True) | (
    Keyword("no", _IDENTBODYCHARS)
    | Keyword("false", _IDENTBODYCHARS)
    | Keyword("off", _IDENTBODYCHARS)
).set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number[0, 7] + Literal("]").suppress()
).set_parse_action(lambda tks: DimensionSet(*tks))
_TENSOR = common.ieee_float | (
    Literal("(").suppress()
    + Group(common.ieee_float[3] | common.ieee_float[6] | common.ieee_float[9])
    + Literal(")").suppress()
).add_parse_action(lambda tks: np.array(tks[0], dtype=float))
_BALANCED = Forward()
_BALANCED <<= Opt(CharsNotIn("()")) + Opt(
    Literal("(") + _BALANCED + Literal(")") + _BALANCED
)
_IDENTIFIER = Combine(
    Word(_IDENTCHARS, _IDENTBODYCHARS) + Opt(Literal("(") + _BALANCED + Literal(")"))
)

_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (Keyword("uniform", _IDENTBODYCHARS).suppress() + _TENSOR) | (
    Keyword("nonuniform", _IDENTBODYCHARS).suppress()
    + (
        (
            Opt(
                Literal("List") + Literal("<") + Literal("scalar") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(dtype=float, ignore=_COMMENT, empty_ok=True)
                | _binary_numeric_list(dtype=np.float64, empty_ok=True)
                | _binary_numeric_list(dtype=np.float32, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("vector") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=3, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=3, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=3, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("symmTensor") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=6, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=6, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=6, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("tensor") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=9, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=9, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=9, empty_ok=True)
            )
        )
    )
)

_DIRECTIVE = Word("#", _IDENTBODYCHARS)
_TOKEN = dbl_quoted_string | _DIRECTIVE | _IDENTIFIER
_DATA = Forward()
_DATA_ENTRY = Forward()
_KEYWORD_ENTRY = _keyword_entry_of(
    _TOKEN | _list_of(_IDENTIFIER),
    Opt(_DATA, default=""),
    directive=_DIRECTIVE,
    data_entry=_DATA_ENTRY,
)
_DICT = _dict_of(_TOKEN, Opt(_DATA, default=""))
_LIST_ENTRY = _DICT | _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = _list_of(_LIST_ENTRY)
_NUMBER = (
    common.number
    | CaselessKeyword("nan").set_parse_action(lambda: np.nan)
    | (CaselessKeyword("inf") | CaselessKeyword("infinity")).set_parse_action(
        lambda: np.inf
    )
    | (CaselessKeyword("-inf") | CaselessKeyword("-infinity")).set_parse_action(
        lambda: -np.inf
    )
)
_DATA_ENTRY <<= _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | _NUMBER | _SWITCH | _TOKEN

_DATA <<= _DATA_ENTRY[1, ...].set_parse_action(
    lambda tks: [tuple(tks)] if len(tks) > 1 else [tks[0]]
)

_STANDALONE_DATA = (
    _ascii_numeric_list(dtype=int, ignore=_COMMENT)
    | _ascii_face_list(ignore=_COMMENT)
    | _ascii_numeric_list(dtype=float, nested=3, ignore=_COMMENT)
    | (
        _binary_numeric_list(dtype=np.int64) + Opt(_binary_numeric_list(dtype=np.int64))
    ).add_parse_action(lambda tks: tuple(tks) if len(tks) > 1 else tks[0])
    | (
        _binary_numeric_list(dtype=np.int32) + Opt(_binary_numeric_list(dtype=np.int32))
    ).add_parse_action(lambda tks: tuple(tks) if len(tks) > 1 else tks[0])
    | _binary_numeric_list(dtype=np.float64, nested=3)
    | _binary_numeric_list(dtype=np.float32, nested=3)
    | _DATA
).add_parse_action(lambda tks: [None, tks[0]])


_FILE = (
    Dict(_KEYWORD_ENTRY[...] + Opt(Group(_STANDALONE_DATA)) + _KEYWORD_ENTRY[...])
    .ignore(_COMMENT)
    .parse_with_tabs()
)

_DATA_OR_DICT = (_DATA | _DICT).ignore(_COMMENT).parse_with_tabs()


@overload
def loads(s: bytes | str, *, keywords: tuple[()]) -> File | StandaloneData: ...


@overload
def loads(
    s: bytes | str, *, keywords: tuple[str, ...] | None = None
) -> File | StandaloneData | Data | SubDict: ...


def loads(
    s: bytes | str, *, keywords: tuple[str, ...] | None = None
) -> File | StandaloneData | Data | SubDict:
    if isinstance(s, bytes):
        s = s.decode("latin-1")

    if keywords == ():
        data = _FILE.parse_string(s, parse_all=True).as_dict()

        if len(data) == 1 and None in data:
            data = data[None]

    else:
        data = _DATA_OR_DICT.parse_string(s, parse_all=True)[0]

    return data


_LOCATED_KEYWORD_ENTRIES = Group(
    _keyword_entry_of(
        _TOKEN,
        Opt(_DATA, default=""),
        directive=_DIRECTIVE,
        data_entry=_DATA_ENTRY,
        located=True,
    )
)[...]
_LOCATED_STANDALONE_DATA = Group(Located(_STANDALONE_DATA))

_LOCATED_FILE = (
    Dict(
        _LOCATED_KEYWORD_ENTRIES
        + Opt(_LOCATED_STANDALONE_DATA)
        + _LOCATED_KEYWORD_ENTRIES
    )
    .ignore(_COMMENT)
    .parse_with_tabs()
)


class Parsed(Mapping[str, Union[Data, StandaloneData, EllipsisType, "Parsed"]]):
    def __init__(self, contents: bytes, *, _parent: "Parsed | None" = None, _key_path: tuple[str, ...] = ()) -> None:
        if _parent is None:
            # Root instance
            self._root_data: MutableMapping[
                tuple[str, ...],
                tuple[int, Data | StandaloneData | EllipsisType, int],
            ] = {}
            for parse_result in _LOCATED_FILE.parse_string(
                contents.decode("latin-1"), parse_all=True
            ):
                self._root_data.update(self._flatten_result(parse_result))

            self.contents = contents
            self.modified = False
            self._parent = None
            self._key_path = ()
        else:
            # Sub-dict instance
            self._root_data = _parent._root_data
            self.contents = _parent.contents
            self.modified = _parent.modified
            self._parent = _parent
            self._key_path = _key_path

    def _get_root(self) -> "Parsed":
        """Get the root Parsed instance."""
        current = self
        while current._parent is not None:
            current = current._parent
        return current

    @staticmethod
    def _flatten_result(
        parse_result: ParseResults, *, _keywords: tuple[str, ...] = ()
    ) -> Mapping[
        tuple[str, ...], tuple[int, Data | StandaloneData | EllipsisType, int]
    ]:
        ret: MutableMapping[
            tuple[str, ...],
            tuple[int, Data | StandaloneData | EllipsisType, int],
        ] = {}
        start = parse_result.locn_start
        assert isinstance(start, int)
        item = parse_result.value
        assert isinstance(item, Sequence)
        end = parse_result.locn_end
        assert isinstance(end, int)
        keyword, *data = item
        if keyword is None:
            assert not _keywords
            assert len(data) == 1
            assert not isinstance(data[0], ParseResults)
            ret[()] = (start, data[0], end)
        else:
            assert isinstance(keyword, str)
            ret[(*_keywords, keyword)] = (start, ..., end)
            for d in data:
                if isinstance(d, ParseResults):
                    ret.update(
                        Parsed._flatten_result(d, _keywords=(*_keywords, keyword))
                    )
                else:
                    ret[(*_keywords, keyword)] = (start, d, end)
        return ret

    def __getitem__(
        self, key: str
    ) -> Data | StandaloneData | EllipsisType | "Parsed":
        full_path = (*self._key_path, key)
        
        # Check if this key exists in the root data
        if full_path not in self._root_data:
            raise KeyError(key)
        
        _, data, _ = self._root_data[full_path]
        
        # If data is Ellipsis (...), this is a sub-dictionary
        if data is ...:
            return Parsed(self.contents, _parent=self._get_root(), _key_path=full_path)
        else:
            return data

    def put(
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        # Handle tuple keys for backward compatibility
        root = self._get_root()
        start, end = root.entry_location(keywords, missing_ok=True)

        diff = len(content) - (end - start)
        for k, (s, d, e) in root._root_data.items():
            if s >= end:
                root._root_data[k] = (s + diff, d, e + diff)
            elif e > start:
                root._root_data[k] = (s, d, e + diff)

        root._root_data[keywords] = (start, data, end + diff)

        root.contents = root.contents[:start] + content + root.contents[end:]
        root.modified = True

        for k in list(root._root_data):
            if keywords != k and keywords == k[: len(keywords)]:
                del root._root_data[k]

    def put_key(
        self,
        key: str,
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        # New method for string keys
        keywords = (*self._key_path, key)
        self.put(keywords, data, content)

    def __delitem__(self, key: str) -> None:
        keywords = (*self._key_path, key)
        root = self._get_root()
        start, end = root.entry_location(keywords)
        del root._root_data[keywords]

        for k in list(root._root_data):
            if keywords == k[: len(keywords)]:
                del root._root_data[k]

        diff = end - start
        for k, (s, d, e) in root._root_data.items():
            if s > end:
                root._root_data[k] = (s - diff, d, e - diff)
            elif e > start:
                root._root_data[k] = (s, d, e - diff)

        root.contents = root.contents[:start] + root.contents[end:]
        root.modified = True

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        keywords = (*self._key_path, key)
        return keywords in self._get_root()._root_data

    def __iter__(self) -> Iterator[str]:
        root = self._get_root()
        keys = set()
        path_len = len(self._key_path)
        
        for k in root._root_data:
            # Check if this key is a direct child of our current path
            if len(k) == path_len + 1 and k[:path_len] == self._key_path:
                keys.add(k[path_len])
        
        return iter(keys)

    def __len__(self) -> int:
        root = self._get_root()
        count = 0
        path_len = len(self._key_path)
        
        for k in root._root_data:
            # Check if this key is a direct child of our current path
            if len(k) == path_len + 1 and k[:path_len] == self._key_path:
                count += 1
        
        return count

    def entry_location(
        self, key: str | tuple[str, ...], *, missing_ok: bool = False
    ) -> tuple[int, int]:
        # Handle both old tuple interface and new string interface for backward compatibility during transition
        if isinstance(key, str):
            keywords = (*self._key_path, key)
        else:
            keywords = key
            
        root = self._get_root()
        try:
            start, _, end = root._root_data[keywords]
        except KeyError:
            if missing_ok:
                if len(keywords) > 1:
                    assert root._root_data.get(keywords[:-1], (None, None, None))[1] is ...
                    start, end = root.entry_location(keywords[:-1])
                    end = root.contents.rindex(b"}", start, end)
                else:
                    end = len(root.contents)

                start = end
            else:
                raise

        return start, end

    def as_dict(self) -> File:
        ret: File = {}
        root = self._get_root()
        
        # Only include keys that are under our current path
        for keywords, (_, data, _) in root._root_data.items():
            # Check if this key is under our current path
            if len(keywords) > len(self._key_path) and keywords[:len(self._key_path)] == self._key_path:
                # Get the relative path from our current position
                relative_keywords = keywords[len(self._key_path):]
                
                r = ret
                for k in relative_keywords[:-1]:
                    if k not in r:
                        r[k] = {}
                    v = r[k]
                    assert isinstance(v, dict)
                    r = cast("File", v)

                assert isinstance(r, dict)
                if relative_keywords:
                    r[relative_keywords[-1]] = {} if data is ... else data

        # Handle the case where we're at a leaf with data but no sub-keys
        if self._key_path in root._root_data:
            _, data, _ = root._root_data[self._key_path]
            if data is not ... and not ret:
                # This is a direct value, not a dict
                return data  # type: ignore[return-value]

        return ret
