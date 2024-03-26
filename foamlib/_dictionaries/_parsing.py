from pyparsing import (
    Dict,
    Forward,
    Group,
    Keyword,
    LineEnd,
    Literal,
    Located,
    Opt,
    QuotedString,
    Word,
    c_style_comment,
    common,
    cpp_style_comment,
    printables,
    identchars,
    identbodychars,
)

from ._values import FoamDimensionSet, FoamDimensioned


_YES = Keyword("yes").set_parse_action(lambda: True)
_NO = Keyword("no").set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamDimensionSet(*tks))
_TOKEN = QuotedString('"', unquote_results=False) | Word(
    identchars + "$", identbodychars
)
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
    ).set_parse_action(lambda tks: [tks[1]] * tks[0])
)
_FIELD = (Keyword("uniform").suppress() + _ITEM) | (
    Keyword("nonuniform").suppress() + _LIST
)
_DIMENSIONED = (Opt(common.identifier) + _DIMENSIONS + _ITEM).set_parse_action(
    lambda tks: FoamDimensioned(*reversed(tks.as_list()))
)
_ITEM <<= (
    _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _YES | _NO | _TOKEN
)

_TOKENS = (
    QuotedString('"', unquote_results=False)
    | Word(printables.replace(";", "").replace("{", "").replace("}", ""))
)[2, ...].set_parse_action(lambda tks: " ".join(tks))

VALUE = (_ITEM ^ _TOKENS).ignore(c_style_comment).ignore(cpp_style_comment)


_UNPARSED_VALUE = (
    QuotedString('"', unquote_results=False)
    | Word(printables.replace(";", "").replace("{", "").replace("}", ""))
)[...]
_KEYWORD = QuotedString('"', unquote_results=False) | Word(
    identchars + "$(,.)", identbodychars + "$(,.)"
)
DICTIONARY = Forward()
_ENTRY = _KEYWORD + (
    (
        Located(_UNPARSED_VALUE).set_parse_action(lambda tks: (tks[0], tks[2]))
        + Literal(";").suppress()
    )
    | (Literal("{").suppress() + DICTIONARY + Literal("}").suppress())
)
DICTIONARY <<= (
    Dict(Group(_ENTRY)[...])
    .set_parse_action(lambda tks: {} if not tks else tks)
    .ignore(c_style_comment)
    .ignore(cpp_style_comment)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore
)
