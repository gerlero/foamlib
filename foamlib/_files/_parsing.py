from __future__ import annotations

import re
import sys
from typing import Tuple, Union, cast

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

from ._types import Data, Dimensioned, DimensionSet, File


def _numeric_list(
    *, nested: int | None = None, ignore: Regex | None = None, force_float: bool = False
) -> ParserElement:
    if not force_float:
        int_pattern = r"(?:-?\d+)"
    float_pattern = r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))"
    spacing_pattern = (
        rf"(?:(?:\s|{ignore.re.pattern})+)" if ignore is not None else r"\s+"
    )

    if nested is None:
        if not force_float:
            int_element_pattern = int_pattern
            int_element = common.integer
        float_element_pattern = float_pattern
        float_element = common.ieee_float
    else:
        if not force_float:
            int_element_pattern = rf"(?:(?:{nested})?{spacing_pattern}?\({spacing_pattern}?(?:{int_pattern}{spacing_pattern}){{{nested - 1}}}{int_pattern}{spacing_pattern}?\))"
            int_element = (
                Opt(Literal(str(nested))).suppress()
                + Literal("(").suppress()
                + Group(common.integer[nested])
                + Literal(")").suppress()
            )
        float_element_pattern = rf"(?:(?:{nested})?{spacing_pattern}?\({spacing_pattern}?(?:{float_pattern}{spacing_pattern}){{{nested - 1}}}{float_pattern}{spacing_pattern}?\))"
        float_element = (
            Opt(Literal(str(nested))).suppress()
            + Literal("(").suppress()
            + Group(common.ieee_float[nested])
            + Literal(")").suppress()
        )

    if not force_float:
        int_list = Forward()
    float_list = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal int_list, float_list

        if not tks:
            count = None
        else:
            (count,) = tks
            assert isinstance(count, int)

        if count is None:
            if not force_float:
                int_list_pattern = rf"\({spacing_pattern}?(?:{int_element_pattern}{spacing_pattern})*{int_element_pattern}{spacing_pattern}?\)"
                float_list_pattern = rf"\({spacing_pattern}?(?:{float_element_pattern}{spacing_pattern})*{float_element_pattern}{spacing_pattern}?\)"
            else:
                float_list_pattern = rf"\({spacing_pattern}?(?:{float_element_pattern}{spacing_pattern})*{float_element_pattern}?{spacing_pattern}?\)"

        elif count == 0:
            if not force_float:
                int_list <<= NoMatch()
                float_list <<= NoMatch()
            else:
                float_list <<= (Literal("(") + Literal(")")).add_parse_action(
                    lambda: np.empty((0, nested) if nested else 0, dtype=float)
                )
            return

        else:
            if not force_float:
                int_list_pattern = rf"\({spacing_pattern}?(?:{int_element_pattern}{spacing_pattern}){{{count - 1}}}{int_element_pattern}{spacing_pattern}?\)"
            float_list_pattern = rf"\({spacing_pattern}?(?:{float_element_pattern}{spacing_pattern}){{{count - 1}}}{float_element_pattern}{spacing_pattern}?\)"

        if not force_float:
            int_list <<= Regex(int_list_pattern).add_parse_action(
                lambda tks: to_array(tks, dtype=int)
            )
        float_list <<= Regex(float_list_pattern).add_parse_action(
            lambda tks: to_array(tks, dtype=float)
        )

    def to_array(
        tks: ParseResults, *, dtype: type
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64 | np.float64]]:
        (s,) = tks
        s = s.replace("(", "").replace(")", "")

        if ignore is not None:
            s = re.sub(ignore.re, " ", s)

        ret: np.ndarray[
            tuple[int] | tuple[int, int], np.dtype[np.int64 | np.float64]
        ] = np.fromstring(s, sep=" ", dtype=dtype)  # type: ignore[assignment]

        if nested is not None:
            ret = ret.reshape(-1, nested)

        return ret

    def to_full_array(
        tks: ParseResults, *, dtype: type
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64 | np.float64]]:
        count, lst = tks
        assert isinstance(count, int)

        if nested is None:
            return np.full(count, lst, dtype=dtype)

        return np.full((count, nested), lst, dtype=dtype)  # type: ignore[return-value]

    count = Opt(common.integer).add_parse_action(process_count)

    ret = count.suppress() + (
        (int_list | float_list) if not force_float else float_list
    )

    if not force_float:
        ret |= (
            common.integer
            + Literal("{").suppress()
            + int_element
            + Literal("}").suppress()
        ).add_parse_action(lambda tks: to_full_array(tks, dtype=int))
    ret |= (
        common.integer
        + Literal("{").suppress()
        + float_element
        + Literal("}").suppress()
    ).add_parse_action(lambda tks: to_full_array(tks, dtype=float))

    if ignore is not None:
        ret.ignore(ignore)

    return ret


def _binary_field(*, nested: int | None = None) -> ParserElement:
    elsize = nested if nested is not None else 1

    binary_field = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal binary_field
        (size,) = tks
        assert isinstance(size, int)

        binary_field <<= Regex(
            rf"\((?s:({'.' * 8 * elsize}|{'.' * 4 * elsize}){{{size}}})\)"
        )

    def to_array(
        tks: ParseResults,
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]:
        size, s = tks
        assert isinstance(size, int)
        assert isinstance(s, str)
        assert s[0] == "("
        assert s[-1] == ")"
        s = s[1:-1]

        float_size = len(s) / elsize / size
        assert float_size in (4, 8)

        dtype = np.float32 if float_size == 4 else float
        ret = np.frombuffer(s.encode("latin-1"), dtype=dtype)

        if nested is not None:
            ret = ret.reshape(-1, nested)

        return ret  # type: ignore[return-value]

    count = common.integer.copy().add_parse_action(process_count)

    return (count + binary_field).add_parse_action(to_array)


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
    | Keyword("y", _IDENTBODYCHARS)
    | Keyword("t", _IDENTBODYCHARS)
).set_parse_action(lambda: True) | (
    Keyword("no", _IDENTBODYCHARS)
    | Keyword("false", _IDENTBODYCHARS)
    | Keyword("off", _IDENTBODYCHARS)
    | Keyword("n", _IDENTBODYCHARS)
    | Keyword("f", _IDENTBODYCHARS)
).set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number[0, 7] + Literal("]").suppress()
).set_parse_action(lambda tks: DimensionSet(*tks))
_TENSOR = common.ieee_float | (
    Literal("(").suppress()
    + Group(common.ieee_float[3] | common.ieee_float[6] | common.ieee_float[9])
    + Literal(")").suppress()
).add_parse_action(lambda tks: np.array(tks[0], dtype=float))
_PARENTHESIZED = Forward()
_IDENTIFIER = Combine(Word(_IDENTCHARS, _IDENTBODYCHARS) + Opt(_PARENTHESIZED))
_PARENTHESIZED <<= Combine(
    Literal("(")
    + (_PARENTHESIZED | Word(_IDENTBODYCHARS) + Opt(_PARENTHESIZED))
    + Literal(")")
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
            + (_numeric_list(force_float=True, ignore=_COMMENT) | _binary_field())
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("vector") + Literal(">")
            ).suppress()
            + (
                _numeric_list(nested=3, force_float=True, ignore=_COMMENT)
                | _binary_field(nested=3)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("symmTensor") + Literal(">")
            ).suppress()
            + (
                _numeric_list(
                    nested=6,
                    force_float=True,
                    ignore=_COMMENT,
                )
                | _binary_field(nested=6)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("tensor") + Literal(">")
            ).suppress()
            + (
                _numeric_list(nested=9, force_float=True, ignore=_COMMENT)
                | _binary_field(nested=9)
            )
        )
    )
)

_DIRECTIVE = Word("#", _IDENTBODYCHARS)
_TOKEN = dbl_quoted_string | _DIRECTIVE | _IDENTIFIER
_DATA = Forward()
_KEYWORD_ENTRY = _keyword_entry_of(_TOKEN | _list_of(_IDENTIFIER), _DATA)
_DICT = _dict_of(_TOKEN, _DATA)
_DATA_ENTRY = Forward()
_LIST_ENTRY = _DICT | _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = (
    _numeric_list(ignore=_COMMENT)
    | _numeric_list(nested=3, ignore=_COMMENT)
    | _numeric_list(nested=4, ignore=_COMMENT)
    | _list_of(_LIST_ENTRY)
)
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

_DATA <<= (
    _DATA_ENTRY[1, ...]
    .set_parse_action(lambda tks: [tuple(tks)] if len(tks) > 1 else [tks[0]])
    .ignore(_COMMENT)
    .parse_with_tabs()
)


def parse_data(s: str) -> Data:
    if not s.strip():
        return ""
    return cast("Data", _DATA.parse_string(s, parse_all=True)[0])


_LOCATED_DICTIONARY = Group(
    _keyword_entry_of(
        _TOKEN,
        Opt(_DATA, default=""),
        directive=_DIRECTIVE,
        data_entry=_DATA_ENTRY,
        located=True,
    )
)[...]
_LOCATED_DATA = Group(Located(_DATA.copy().add_parse_action(lambda tks: ["", tks[0]])))

_FILE = (
    Dict(_LOCATED_DICTIONARY + Opt(_LOCATED_DATA) + _LOCATED_DICTIONARY)
    .ignore(_COMMENT)
    .parse_with_tabs()
)


class Parsed(Mapping[Tuple[str, ...], Union[Data, EllipsisType]]):
    def __init__(self, contents: bytes) -> None:
        self._parsed: MutableMapping[
            tuple[str, ...],
            tuple[int, Data | EllipsisType, int],
        ] = {}
        for parse_result in _FILE.parse_string(
            contents.decode("latin-1"), parse_all=True
        ):
            self._parsed.update(self._flatten_result(parse_result))

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_result(
        parse_result: ParseResults, *, _keywords: tuple[str, ...] = ()
    ) -> Mapping[tuple[str, ...], tuple[int, Data | EllipsisType, int]]:
        ret: MutableMapping[
            tuple[str, ...],
            tuple[int, Data | EllipsisType, int],
        ] = {}
        start = parse_result.locn_start
        assert isinstance(start, int)
        item = parse_result.value
        assert isinstance(item, Sequence)
        end = parse_result.locn_end
        assert isinstance(end, int)
        keyword, *data = item
        assert isinstance(keyword, str)
        if not keyword:
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

    def __getitem__(self, keywords: tuple[str, ...]) -> Data | EllipsisType:
        _, data, _ = self._parsed[keywords]
        return data

    def put(
        self,
        keywords: tuple[str, ...],
        data: Data | EllipsisType,
        content: bytes,
    ) -> None:
        start, end = self.entry_location(keywords, missing_ok=True)

        diff = len(content) - (end - start)
        for k, (s, d, e) in self._parsed.items():
            if s > end:
                self._parsed[k] = (s + diff, d, e + diff)
            elif e > start:
                self._parsed[k] = (s, d, e + diff)

        self._parsed[keywords] = (start, data, end + diff)

        self.contents = self.contents[:start] + content + self.contents[end:]
        self.modified = True

        for k in list(self._parsed):
            if keywords != k and keywords == k[: len(keywords)]:
                del self._parsed[k]

    def __delitem__(self, keywords: tuple[str, ...]) -> None:
        start, end = self.entry_location(keywords)
        del self._parsed[keywords]

        for k in list(self._parsed):
            if keywords == k[: len(keywords)]:
                del self._parsed[k]

        diff = end - start
        for k, (s, d, e) in self._parsed.items():
            if s > end:
                self._parsed[k] = (s - diff, d, e - diff)
            elif e > start:
                self._parsed[k] = (s, d, e - diff)

        self.contents = self.contents[:start] + self.contents[end:]
        self.modified = True

    def __contains__(self, keywords: object) -> bool:
        return keywords in self._parsed

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self._parsed)

    def __len__(self) -> int:
        return len(self._parsed)

    def entry_location(
        self, keywords: tuple[str, ...], *, missing_ok: bool = False
    ) -> tuple[int, int]:
        try:
            start, _, end = self._parsed[keywords]
        except KeyError:
            if missing_ok:
                if len(keywords) > 1:
                    assert self[keywords[:-1]] is ...
                    start, end = self.entry_location(keywords[:-1])
                    end = self.contents.rindex(b"}", start, end)
                else:
                    end = len(self.contents)

                start = end
            else:
                raise

        return start, end

    def as_dict(self) -> File:
        ret: File = {}
        for keywords, (_, data, _) in self._parsed.items():
            r = ret
            for k in keywords[:-1]:
                v = r[k]
                assert isinstance(v, dict)
                r = cast("File", v)

            assert isinstance(r, dict)
            if keywords:
                r[keywords[-1]] = {} if data is ... else data
            else:
                assert data is not ...
                r[None] = data

        return ret
