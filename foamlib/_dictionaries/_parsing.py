import sys
from typing import Optional, Tuple, Union

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, MutableMapping, Sequence
else:
    from typing import Mapping, MutableMapping, Sequence

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

_SWITCH = (
    Keyword("yes") | Keyword("true") | Keyword("on") | Keyword("y") | Keyword("t")
).set_parse_action(lambda: True) | (
    Keyword("no") | Keyword("false") | Keyword("off") | Keyword("n") | Keyword("f")
).set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamDictionaryBase.DimensionSet(*tks))


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

Parsed = Mapping[Sequence[str], Tuple[int, Optional[FoamDictionaryBase.Value], int]]


def _flatten_result(
    parse_result: ParseResults, *, _keywords: Sequence[str] = ()
) -> Parsed:
    ret: MutableMapping[
        Sequence[str], Tuple[int, Optional[FoamDictionaryBase.Value], int]
    ] = {}
    start = parse_result.locn_start
    assert isinstance(start, int)
    item = parse_result.value
    assert isinstance(item, Sequence)
    end = parse_result.locn_end
    assert isinstance(end, int)
    key, *values = item
    assert isinstance(key, str)
    ret[(*_keywords, key)] = (start, None, end)
    for value in values:
        if isinstance(value, ParseResults):
            ret.update(_flatten_result(value, _keywords=(*_keywords, key)))
        else:
            ret[(*_keywords, key)] = (start, value, end)
    return ret


def parse(
    contents: str,
) -> Parsed:
    parse_results = _FILE.parse_string(contents, parse_all=True)
    ret: MutableMapping[
        Sequence[str], Tuple[int, Optional[FoamDictionaryBase.Value], int]
    ] = {}
    for parse_result in parse_results:
        ret.update(_flatten_result(parse_result))
    return ret


def get_value(
    parsed: Parsed,
    keywords: Tuple[str, ...],
) -> Optional[FoamDictionaryBase.Value]:
    """Value of an entry."""
    _, value, _ = parsed[keywords]
    return value


def get_entry_locn(
    parsed: Parsed,
    keywords: Tuple[str, ...],
    *,
    missing_ok: bool = False,
) -> Tuple[int, int]:
    """Location of an entry or where it should be inserted."""
    try:
        start, _, end = parsed[keywords]
    except KeyError:
        if missing_ok:
            if len(keywords) > 1:
                _, _, end = parsed[keywords[:-1]]
                end -= 1
            else:
                end = -1

            start = end
        else:
            raise

    return start, end


def as_dict(parsed: Parsed) -> FoamDictionaryBase._Dict:
    """Return a nested dict representation of the file."""
    ret: FoamDictionaryBase._Dict = {}
    for keywords, (_, value, _) in parsed.items():
        r = ret
        for k in keywords[:-1]:
            assert isinstance(r, dict)
            v = r[k]
            assert isinstance(v, dict)
            r = v

        assert isinstance(r, dict)
        r[keywords[-1]] = {} if value is None else value

    return ret
