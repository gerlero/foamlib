import sys
from typing import Tuple, Union

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    from typing import Any as EllipsisType

from pyparsing import (
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
    cpp_style_comment,
    identbodychars,
    printables,
)

from ._base import FoamDictionaryBase


def _list_of(elem: ParserElement) -> ParserElement:
    return Opt(
        Literal("List") + Literal("<") + common.identifier + Literal(">")
    ).suppress() + (
        (
            Opt(common.integer).suppress()
            + (
                Literal("(").suppress()
                + Group((elem)[...], aslist=True)
                + Literal(")").suppress()
            )
        )
        | (
            common.integer + Literal("{").suppress() + elem + Literal("}").suppress()
        ).set_parse_action(lambda tks: [[tks[1]] * tks[0]])
    )


def _dictionary_of(
    keyword: ParserElement,
    value: ParserElement,
    *,
    len: Union[int, EllipsisType] = ...,
    located: bool = False,
) -> ParserElement:
    subdict = Forward()

    entry = keyword + (
        (Literal("{").suppress() + subdict + Literal("}").suppress())
        | (value + Literal(";").suppress())
    )

    if located:
        entry = Located(entry)

    subdict <<= Dict(Group(entry)[...], asdict=not located)

    return Dict(Group(entry)[len], asdict=not located)


_SWITCH = (
    Keyword("yes") | Keyword("true") | Keyword("on") | Keyword("y") | Keyword("t")
).set_parse_action(lambda: True) | (
    Keyword("no") | Keyword("false") | Keyword("off") | Keyword("n") | Keyword("f")
).set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamDictionaryBase.DimensionSet(*tks))

_TENSOR = _list_of(common.number) | common.number
_IDENTIFIER = Word(identbodychars + "$", printables.replace(";", ""))
_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: FoamDictionaryBase.Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (Keyword("uniform").suppress() + _TENSOR) | (
    Keyword("nonuniform").suppress() + _list_of(_TENSOR)
)
_TOKEN = QuotedString('"', unquote_results=False) | _IDENTIFIER
_ITEM = Forward()
_ENTRY = _dictionary_of(_IDENTIFIER, _ITEM, len=1)
_LIST = _list_of(_ENTRY | _ITEM)
_ITEM <<= _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _SWITCH | _TOKEN

_TOKENS = (
    QuotedString('"', unquote_results=False) | Word(printables.replace(";", ""))
)[2, ...].set_parse_action(lambda tks: " ".join(tks))

_VALUE = _ITEM ^ _TOKENS

_FILE = (
    _dictionary_of(_TOKEN, Opt(_VALUE, default=""), located=True)
    .ignore(c_style_comment)
    .ignore(cpp_style_comment)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore [no-untyped-call]
)


class Parsed(Mapping[Tuple[str, ...], Union[FoamDictionaryBase.Value, EllipsisType]]):
    def __init__(self, contents: str) -> None:
        self._parsed: MutableMapping[
            Tuple[str, ...],
            Tuple[int, Union[FoamDictionaryBase.Value, EllipsisType], int],
        ] = {}
        for parse_result in _FILE.parse_string(contents, parse_all=True):
            self._parsed.update(self._flatten_result(parse_result))

    @staticmethod
    def _flatten_result(
        parse_result: ParseResults, *, _keywords: Tuple[str, ...] = ()
    ) -> Mapping[
        Tuple[str, ...], Tuple[int, Union[FoamDictionaryBase.Value, EllipsisType], int]
    ]:
        ret: MutableMapping[
            Tuple[str, ...],
            Tuple[int, Union[FoamDictionaryBase.Value, EllipsisType], int],
        ] = {}
        start = parse_result.locn_start
        assert isinstance(start, int)
        item = parse_result.value
        assert isinstance(item, Sequence)
        end = parse_result.locn_end
        assert isinstance(end, int)
        key, *values = item
        assert isinstance(key, str)
        ret[(*_keywords, key)] = (start, ..., end)
        for value in values:
            if isinstance(value, ParseResults):
                ret.update(Parsed._flatten_result(value, _keywords=(*_keywords, key)))
            else:
                ret[(*_keywords, key)] = (start, value, end)
        return ret

    def __getitem__(
        self, keywords: Tuple[str, ...]
    ) -> Union[FoamDictionaryBase.Value, EllipsisType]:
        _, value, _ = self._parsed[keywords]
        return value

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
                    end = -1

                start = end
            else:
                raise

        return start, end

    def as_dict(self) -> FoamDictionaryBase._Dict:
        ret: FoamDictionaryBase._Dict = {}
        for keywords, (_, value, _) in self._parsed.items():
            r = ret
            for k in keywords[:-1]:
                assert isinstance(r, dict)
                v = r[k]
                assert isinstance(v, dict)
                r = v

            assert isinstance(r, dict)
            r[keywords[-1]] = {} if value is ... else value  # type: ignore [assignment]

        return ret
