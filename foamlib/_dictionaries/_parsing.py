from typing import cast

from pyparsing import (
    Forward,
    Group,
    Keyword,
    Literal,
    Opt,
    QuotedString,
    Word,
    common,
    printables,
)

from ._values import FoamValue, FoamDimensionSet, FoamDimensioned


_YES = Keyword("yes").set_parse_action(lambda s, loc, tks: True)
_NO = Keyword("no").set_parse_action(lambda s, loc, tks: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda s, loc, tks: FoamDimensionSet(*tks))
_TOKEN = common.identifier | QuotedString('"', unquote_results=False)
_ITEM = Forward()
_LIST = Opt(
    Literal("List") + Literal("<") + common.identifier + Literal(">")
).suppress() + (
    (
        Opt(common.integer).suppress()
        + Literal("(").suppress()
        + Group(_ITEM[...])
        + Literal(")").suppress()
    )
    | (
        common.integer + Literal("{").suppress() + _ITEM + Literal("}").suppress()
    ).set_parse_action(lambda s, loc, tks: [tks[1]] * tks[0])
)
_FIELD = (Keyword("uniform").suppress() + _ITEM) | (
    Keyword("nonuniform").suppress() + _LIST
)
_DIMENSIONED = (Opt(common.identifier) + _DIMENSIONS + _ITEM).set_parse_action(
    lambda s, loc, tks: FoamDimensioned(*reversed(tks.as_list()))
)
_ITEM <<= (
    _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _YES | _NO | _TOKEN
)

_TOKENS = (QuotedString('"', unquote_results=False) | Word(printables))[
    1, ...
].set_parse_action(lambda s, loc, tks: " ".join(tks))

_VALUE = _ITEM ^ _TOKENS


def parse(value: str) -> FoamValue:
    return cast(FoamValue, _VALUE.parse_string(value, parse_all=True).as_list()[0])
