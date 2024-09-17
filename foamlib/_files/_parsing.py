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
) -> Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]:
    elsize = len(tks[0]) // 8

    arr = array.array("d", "".join(tks).encode("latin-1"))

    all: Union[Sequence[float], Sequence[Sequence[float]]]

    if elsize != 1:
        all = [arr[i : i + elsize].tolist() for i in range(0, len(arr), elsize)]
    else:
        all = arr.tolist()

    return [all]


_BINARY_FIELD = (
    Keyword("nonuniform").suppress()
    + Literal("List").suppress()
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


_SWITCH = (
    Keyword("yes") | Keyword("true") | Keyword("on") | Keyword("y") | Keyword("t")
).set_parse_action(lambda: True) | (
    Keyword("no") | Keyword("false") | Keyword("off") | Keyword("n") | Keyword("f")
).set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamFileBase.DimensionSet(*tks))
_TENSOR = _list_of(common.number) | common.number
_IDENTIFIER = Combine(
    Word(identchars + "$", printables, exclude_chars="{(;)}")
    + Opt(Literal("(") + Word(printables, exclude_chars="{(;)}") + Literal(")"))
)
_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: FoamFileBase.Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (
    (Keyword("uniform").suppress() + _TENSOR)
    | (Keyword("nonuniform").suppress() + _list_of(_TENSOR))
    | _BINARY_FIELD
)
_TOKEN = QuotedString('"', unquote_results=False) | _IDENTIFIER
_DATA = Forward()
_KEYWORD = Combine(Literal("(") + Word(identchars + " ") + Literal(")")) | _TOKEN
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


class Parsed(Mapping[Tuple[str, ...], Union[FoamFileBase.Data, EllipsisType]]):
    def __init__(self, contents: bytes) -> None:
        self._parsed: MutableMapping[
            Tuple[str, ...],
            Tuple[int, Union[FoamFileBase.Data, EllipsisType], int],
        ] = {}
        self._end = len(contents)

        for parse_result in _FILE.parse_string(
            contents.decode("latin-1"), parse_all=True
        ):
            self._parsed.update(self._flatten_result(parse_result))

    @staticmethod
    def _flatten_result(
        parse_result: ParseResults, *, _keywords: Tuple[str, ...] = ()
    ) -> Mapping[
        Tuple[str, ...], Tuple[int, Union[FoamFileBase.Data, EllipsisType], int]
    ]:
        ret: MutableMapping[
            Tuple[str, ...],
            Tuple[int, Union[FoamFileBase.Data, EllipsisType], int],
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
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union[FoamFileBase.Data, EllipsisType]:
        if isinstance(keywords, str):
            keywords = (keywords,)

        _, data, _ = self._parsed[keywords]
        return data

    def __contains__(self, keywords: object) -> bool:
        return keywords in self._parsed

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self._parsed)

    def __len__(self) -> int:
        return len(self._parsed)

    def entry_location(
        self, keywords: Tuple[str, ...], *, missing_ok: bool = False
    ) -> Tuple[int, int]:
        try:
            start, _, end = self._parsed[keywords]
        except KeyError:
            if missing_ok:
                if len(keywords) > 1:
                    _, _, end = self._parsed[keywords[:-1]]
                    end -= 1
                else:
                    end = self._end

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
