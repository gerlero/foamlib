from __future__ import annotations

import numpy as np
from pyparsing import (
    CaselessKeyword,
    CharsNotIn,
    Combine,
    Dict,
    Forward,
    Group,
    Keyword,
    Literal,
    Located,
    Opt,
    Regex,
    Word,
    common,
    dbl_quoted_string,
    identchars,
    printables,
)

from ..types import Dimensioned, DimensionSet
from ._elements import (
    ASCIIFacesLikeList,
    ASCIINumericList,
    MatchLongest,
    binary_numeric_list,
    dict_of,
    keyword_entry_of,
    list_of,
)

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
).add_parse_action(lambda tks: np.array(tks[0]))
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
                ASCIINumericList(dtype=float, empty_ok=True)
                | binary_numeric_list(dtype=np.float64, empty_ok=True)
                | binary_numeric_list(dtype=np.float32, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("vector") + Literal(">")
            ).suppress()
            + (
                ASCIINumericList(dtype=float, elshape=(3,), empty_ok=True)
                | binary_numeric_list(np.float64, elshape=(3,), empty_ok=True)
                | binary_numeric_list(np.float32, elshape=(3,), empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("symmTensor") + Literal(">")
            ).suppress()
            + (
                ASCIINumericList(dtype=float, elshape=(6,), empty_ok=True)
                | binary_numeric_list(np.float64, elshape=(6,), empty_ok=True)
                | binary_numeric_list(np.float32, elshape=(6,), empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("tensor") + Literal(">")
            ).suppress()
            + (
                ASCIINumericList(dtype=float, elshape=(9,), empty_ok=True)
                | binary_numeric_list(np.float64, elshape=(9,), empty_ok=True)
                | binary_numeric_list(np.float32, elshape=(9,), empty_ok=True)
            )
        )
    )
)

_DIRECTIVE = Word("#", _IDENTBODYCHARS)
_KEYWORD = dbl_quoted_string | _IDENTIFIER
_TOKEN = _KEYWORD | _DIRECTIVE
_DATA = Forward()
_DATA_ENTRY = Forward()
_KEYWORD_ENTRY = keyword_entry_of(
    _KEYWORD | list_of(_IDENTIFIER),
    Opt(_DATA, default=""),
    directive=_DIRECTIVE,
    data_entry=_DATA_ENTRY,
)
_DICT = dict_of(_KEYWORD, Opt(_DATA, default=""))
_LIST_ENTRY = _DICT | _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = list_of(_LIST_ENTRY)
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
    ASCIINumericList(dtype=int)
    | ASCIIFacesLikeList()
    | ASCIINumericList(dtype=float, elshape=(3,))
    | MatchLongest(
        (
            (
                binary_numeric_list(dtype=np.int32)
                + Opt(binary_numeric_list(dtype=np.int32))
            ).add_parse_action(lambda tks: tuple(tks) if len(tks) > 1 else tks[0])
            | binary_numeric_list(dtype=np.float64)
            | binary_numeric_list(dtype=np.float64, elshape=(3,))
            | binary_numeric_list(dtype=np.float32, elshape=(3,)),
            _DATA,
        )
    )
).add_parse_action(lambda tks: [None, tks[0]])


_LOCATED_KEYWORD_ENTRIES = Group(
    keyword_entry_of(
        _KEYWORD,
        Opt(_DATA, default=""),
        directive=_DIRECTIVE,
        data_entry=_DATA_ENTRY,
        located=True,
    )
)[...]
_LOCATED_STANDALONE_DATA = Group(Located(_STANDALONE_DATA))

FILE = (
    Dict(
        _LOCATED_KEYWORD_ENTRIES
        + Opt(_LOCATED_STANDALONE_DATA)
        + _LOCATED_KEYWORD_ENTRIES
    )
    .ignore(_COMMENT)
    .parse_with_tabs()
)
