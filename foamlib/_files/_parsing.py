from __future__ import annotations

import array
import sys
from typing import Tuple, Union, cast

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    from typing import Any as EllipsisType

from pyparsing import (
    CharsNotIn,
    Combine,
    Dict,
    Forward,
    Group,
    Keyword,
    LineEnd,
    Literal,
    Located,
    Opt,
    ParserElement,
    ParseResults,
    QuotedString,
    Word,
    c_style_comment,
    common,
    counted_array,
    cpp_style_comment,
    identchars,
    printables,
)

from ._base import FoamFileBase


def _list_of(entry: ParserElement) -> ParserElement:
    return Opt(
        Literal("List") + Literal("<") + common.identifier + Literal(">")
    ).suppress() + (
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


def _keyword_entry_of(
    keyword: ParserElement,
    data_entries: ParserElement,
    *,
    located: bool = False,
) -> ParserElement:
    subdict = Forward()

    keyword_entry = keyword + (
        (Literal("{").suppress() + subdict + Literal("}").suppress())
        | (data_entries + Literal(";").suppress())
    )

    if located:
        keyword_entry = Located(keyword_entry)

    subdict <<= Dict(Group(keyword_entry)[...], asdict=not located)

    return keyword_entry


def _unpack_binary_field(
    tks: ParseResults,
) -> Sequence[Sequence[float] | Sequence[Sequence[float]]]:
    elsize = len(tks[0]) // 8

    arr = array.array("d", "".join(tks).encode("latin-1"))

    values: Sequence[float] | Sequence[Sequence[float]]

    if elsize != 1:
        values = [arr[i : i + elsize].tolist() for i in range(0, len(arr), elsize)]
    else:
        values = arr.tolist()

    return [values]


_IDENTCHARS = identchars + "$"
_IDENTBODYCHARS = (
    printables.replace(";", "")
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
).set_parse_action(lambda tks: FoamFileBase.DimensionSet(*tks))
_TENSOR = _list_of(common.number) | common.number
_IDENTIFIER = Combine(
    Word(_IDENTCHARS, _IDENTBODYCHARS, exclude_chars="()")
    + Opt(Literal("(") + Word(_IDENTBODYCHARS, exclude_chars="()") + Literal(")"))
)
_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: FoamFileBase.Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (Keyword("uniform").suppress() + _TENSOR) | (
    Keyword("nonuniform").suppress()
    + (
        _list_of(_TENSOR)
        | (
            Literal("List").suppress()
            + Literal("<").suppress()
            + (
                counted_array(
                    CharsNotIn(exact=8),
                    Literal("scalar").suppress()
                    + Literal(">").suppress()
                    + common.integer
                    + Literal("(").suppress(),
                )
                | counted_array(
                    CharsNotIn(exact=8 * 3),
                    Literal("vector").suppress()
                    + Literal(">").suppress()
                    + common.integer
                    + Literal("(").suppress(),
                )
                | counted_array(
                    CharsNotIn(exact=8 * 6),
                    Literal("symmTensor").suppress()
                    + Literal(">").suppress()
                    + common.integer
                    + Literal("(").suppress(),
                )
                | counted_array(
                    CharsNotIn(exact=8 * 9),
                    Literal("tensor").suppress()
                    + Literal(">").suppress()
                    + common.integer
                    + Literal("(").suppress(),
                )
            )
            + Literal(")").suppress()
        ).set_parse_action(_unpack_binary_field)
    )
)
_TOKEN = QuotedString('"', unquote_results=False) | _IDENTIFIER
_DATA = Forward()
_KEYWORD = _TOKEN | _list_of(_IDENTIFIER).set_parse_action(
    lambda tks: "(" + " ".join(tks[0]) + ")"
)
_KEYWORD_ENTRY = Dict(Group(_keyword_entry_of(_KEYWORD, _DATA)), asdict=True)
_DATA_ENTRY = Forward()
_LIST_ENTRY = _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = _list_of(_LIST_ENTRY)
_DATA_ENTRY <<= (
    _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _SWITCH | _TOKEN
)

_DATA <<= _DATA_ENTRY[1, ...].set_parse_action(
    lambda tks: tuple(tks) if len(tks) > 1 else [tks[0]]
)

_FILE = (
    Dict(
        Group(_keyword_entry_of(_KEYWORD, Opt(_DATA, default=""), located=True))[...]
        + Opt(
            Group(
                Located(
                    _DATA_ENTRY[1, ...].set_parse_action(
                        lambda tks: [None, tuple(tks) if len(tks) > 1 else tks[0]]
                    )
                )
            )
        )
        + Group(_keyword_entry_of(_KEYWORD, Opt(_DATA, default=""), located=True))[...]
    )
    .ignore(c_style_comment)
    .ignore(cpp_style_comment)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore [no-untyped-call]
    .parse_with_tabs()
)


class Parsed(Mapping[Tuple[str, ...], Union[FoamFileBase._DataEntry, EllipsisType]]):
    def __init__(self, contents: bytes) -> None:
        self._parsed: MutableMapping[
            tuple[str, ...],
            tuple[int, FoamFileBase._DataEntry | EllipsisType, int],
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
    ) -> Mapping[
        tuple[str, ...], tuple[int, FoamFileBase._DataEntry | EllipsisType, int]
    ]:
        ret: MutableMapping[
            tuple[str, ...],
            tuple[int, FoamFileBase._DataEntry | EllipsisType, int],
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
        self, keywords: tuple[str, ...]
    ) -> FoamFileBase._DataEntry | EllipsisType:
        _, data, _ = self._parsed[keywords]
        return data

    def put(
        self,
        keywords: tuple[str, ...],
        data: FoamFileBase._DataEntry | EllipsisType,
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

    def as_dict(self) -> FoamFileBase._File:
        ret: FoamFileBase._File = {}
        for keywords, (_, data, _) in self._parsed.items():
            r = ret
            for k in keywords[:-1]:
                v = r[k]
                assert isinstance(v, dict)
                r = cast(FoamFileBase._File, v)

            assert isinstance(r, dict)
            if keywords:
                r[keywords[-1]] = {} if data is ... else data
            else:
                assert data is not ...
                r[None] = data

        return ret
