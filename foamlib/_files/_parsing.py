from __future__ import annotations

import dataclasses
import re
import sys
from typing import TYPE_CHECKING, cast

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Iterator, Sequence
else:
    from typing import Collection, Iterator, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(...)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping, with_default
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
    ParseException,
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

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

from ._util import add_to_mapping, as_dict_check_unique
from .types import Dimensioned, DimensionSet

if TYPE_CHECKING:
    from ._typing import Data, File, StandaloneData, SubDict


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
        Literal("{").suppress() + Group(keyword_entry)[...] + Literal("}").suppress()
    )

    if not located:
        dict_.set_parse_action(lambda tks: as_dict_check_unique(tks.as_list()))

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
_KEYWORD = dbl_quoted_string | _IDENTIFIER
_TOKEN = _KEYWORD | _DIRECTIVE
_DATA = Forward()
_DATA_ENTRY = Forward()
_KEYWORD_ENTRY = _keyword_entry_of(
    _KEYWORD | _list_of(_IDENTIFIER),
    Opt(_DATA, default=""),
    directive=_DIRECTIVE,
    data_entry=_DATA_ENTRY,
)
_DICT = _dict_of(_KEYWORD, Opt(_DATA, default=""))
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


_LOCATED_KEYWORD_ENTRIES = Group(
    _keyword_entry_of(
        _KEYWORD,
        Opt(_DATA, default=""),
        directive=_DIRECTIVE,
        data_entry=_DATA_ENTRY,
        located=True,
    )
)[...]
_LOCATED_STANDALONE_DATA = Group(Located(_STANDALONE_DATA))

_FILE = (
    Dict(
        _LOCATED_KEYWORD_ENTRIES
        + Opt(_LOCATED_STANDALONE_DATA)
        + _LOCATED_KEYWORD_ENTRIES
    )
    .ignore(_COMMENT)
    .parse_with_tabs()
)


class Parsed(
    MutableMultiMapping["tuple[str, ...]", "Data | StandaloneData | EllipsisType"]
):
    @dataclasses.dataclass
    class _Entry:
        data: Data | StandaloneData | EllipsisType
        start: int
        end: int

    def __init__(self, contents: bytes | str) -> None:
        if isinstance(contents, bytes):
            contents_str = contents.decode("latin-1")
        else:
            contents_str = contents
            contents = contents.encode("latin-1")

        try:
            parse_results = _FILE.parse_string(contents_str, parse_all=True)
        except ParseException as e:
            msg = f"Failed to parse contents: {e}"
            raise ValueError(msg) from e
        self._parsed = self._flatten_results(parse_results)

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_results(
        parse_results: ParseResults | Sequence[ParseResults],
        *,
        _keywords: tuple[str, ...] = (),
    ) -> MultiDict[tuple[str, ...], Parsed._Entry]:
        ret: MultiDict[tuple[str, ...], Parsed._Entry] = MultiDict()
        for parse_result in parse_results:
            value = parse_result.value
            assert isinstance(value, Sequence)
            start = parse_result.locn_start
            assert isinstance(start, int)
            end = parse_result.locn_end
            assert isinstance(end, int)
            keyword, *data = value
            if keyword is None:
                assert not _keywords
                assert len(data) == 1
                assert not isinstance(data[0], ParseResults)
                assert () not in ret
                ret[()] = Parsed._Entry(data[0], start, end)
            else:
                assert isinstance(keyword, str)
                if len(data) == 0 or isinstance(data[0], ParseResults):
                    if (*_keywords, keyword) in ret:
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret[(*_keywords, keyword)] = Parsed._Entry(..., start, end)
                    ret.extend(
                        Parsed._flatten_results(data, _keywords=(*_keywords, keyword))
                    )
                else:
                    if (*_keywords, keyword) in ret and not keyword.startswith("#"):
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret.add((*_keywords, keyword), Parsed._Entry(data[0], start, end))
        return ret

    @override
    @with_default
    def getall(
        self, keywords: tuple[str, ...]
    ) -> Collection[Data | StandaloneData | EllipsisType]:
        return [entry.data for entry in self._parsed.getall(keywords)]

    @override
    def __setitem__(
        self, key: tuple[str, ...], value: Data | StandaloneData | EllipsisType
    ) -> None:  # pragma: no cover
        msg = "Use 'put' method instead"
        raise NotImplementedError(msg)

    def put(
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        start, end = self.entry_location(keywords)

        self._update_content(start, end, content)
        self._parsed[keywords] = Parsed._Entry(data, start, start + len(content))
        self._remove_child_entries(keywords)

    @override
    def add(  # type: ignore[override]
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        if keywords:
            if keywords in self._parsed and not keywords[-1].startswith("#"):
                msg = f"Cannot add duplicate non-directive entry: {keywords}"
                raise ValueError(msg)
            if keywords[-1].startswith("#") and data is ...:
                msg = f"Cannot add sub-dictionary with name: {keywords[-1]}"
                raise ValueError(msg)

        start, end = self.entry_location(keywords, add=True)

        self._parsed.add(keywords, Parsed._Entry(data, start, end))
        self._update_content(start, end, content)

    @override
    @with_default
    def popone(self, keywords: tuple[str, ...]) -> Data | StandaloneData | EllipsisType:
        start, end = self.entry_location(keywords)
        entry = self._parsed.popone(keywords)
        self._remove_child_entries(keywords)
        self._update_content(start, end, b"")
        return entry.data

    @override
    def __contains__(self, keywords: object) -> bool:
        return keywords in self._parsed

    @override
    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self._parsed)

    @override
    def __len__(self) -> int:
        return len(self._parsed)

    def _update_content(self, start: int, end: int, new_content: bytes) -> None:
        """Update content and adjust positions of other entries."""
        diff = len(new_content) - (end - start)

        # Update positions of other entries if content length changed
        if diff != 0:
            for entry in self._parsed.values():
                assert isinstance(entry, Parsed._Entry)
                if entry.start >= end:
                    entry.start += diff
                    entry.end += diff
                elif entry.end > start:
                    entry.end += diff

        self.contents = self.contents[:start] + new_content + self.contents[end:]
        self.modified = True

    def _remove_child_entries(self, keywords: tuple[str, ...]) -> None:
        """Remove all child entries of the given keywords."""
        for k in list(self._parsed):
            if keywords != k and keywords == k[: len(keywords)]:
                del self._parsed[k]

    def entry_location(
        self, keywords: tuple[str, ...], *, add: bool = False
    ) -> tuple[int, int]:
        if add or keywords not in self._parsed:
            if len(keywords) > 1:
                assert self[keywords[:-1]] is ...
                start, end = self.entry_location(keywords[:-1])
                end = self.contents.rindex(b"}", start, end)
            else:
                end = len(self.contents)

            start = end
        else:
            entry = self._parsed[keywords]
            start = entry.start
            end = entry.end

        return start, end

    def as_dict(self) -> File:
        ret: File = {}
        for keywords, entry in self._parsed.items():
            assert isinstance(entry, Parsed._Entry)
            if not keywords:
                assert entry.data is not ...
                assert None not in ret
                ret[None] = entry.data
            elif entry.data is ...:
                parent: File | SubDict = ret
                for k in keywords[:-1]:
                    sub = parent[k]
                    assert isinstance(sub, (dict, MultiDict))
                    parent = sub
                assert keywords[-1] not in parent
                parent[keywords[-1]] = {}
            else:
                assert entry.data is not ...
                if len(keywords) == 1:
                    ret = add_to_mapping(ret, keywords[0], entry.data)
                else:
                    grandparent: File | SubDict = ret
                    for k in keywords[:-2]:
                        sub = grandparent[k]
                        assert isinstance(sub, (dict, MultiDict))
                        grandparent = sub
                    sub = grandparent[keywords[-2]]
                    assert isinstance(sub, (dict, MultiDict))
                    grandparent[keywords[-2]] = add_to_mapping(
                        sub, keywords[-1], cast("Data", entry.data)
                    )

        return ret
