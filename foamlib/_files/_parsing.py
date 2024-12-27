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
    Regex,
    Word,
    common,
    counted_array,
    dbl_quoted_string,
    identchars,
    printables,
)

from ._types import Data, Dimensioned, DimensionSet, File, TensorKind


def _tensor(tensor_kind: TensorKind, *, ignore: Regex | None = None) -> Regex:
    float_pattern = r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))"

    if tensor_kind == TensorKind.SCALAR:
        ret = Regex(float_pattern)
        ret.add_parse_action(lambda tks: [float(tks[0])])
        return ret

    ignore_pattern = rf"(?:\s|{ignore.re.pattern})+" if ignore is not None else r"\s+"

    ret = Regex(
        rf"\((?:{ignore_pattern})?(?:{float_pattern}{ignore_pattern}){{{tensor_kind.size - 1}}}{float_pattern}(?:{ignore_pattern})?\)"
    )
    ret.add_parse_action(
        lambda tks: np.fromstring(
            re.sub(ignore.re, " ", tks[0][1:-1])
            if ignore is not None
            else tks[0][1:-1],
            sep=" ",
        )
    )
    return ret


def _list_of(entry: ParserElement) -> ParserElement:
    return Opt(
        Literal("List") + Literal("<") + _IDENTIFIER + Literal(">")
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


def _parse_ascii_field(
    s: str, tensor_kind: TensorKind, *, ignore: Regex | None
) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64]]:
    if ignore is not None:
        s = re.sub(ignore.re, " ", s)
    s = s.replace("(", " ").replace(")", " ")

    return np.fromstring(s, sep=" ").reshape(-1, *tensor_kind.shape)  # type: ignore [return-value]


def _unpack_binary_field(
    b: bytes, tensor_kind: TensorKind, *, length: int
) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]:
    float_size = len(b) / tensor_kind.size / length
    assert float_size in (4, 8)

    dtype = np.float32 if float_size == 4 else float
    return np.frombuffer(b, dtype=dtype).reshape(-1, *tensor_kind.shape)  # type: ignore [return-value]


def _tensor_list(
    tensor_kind: TensorKind, *, ignore: Regex | None = None
) -> ParserElement:
    tensor = _tensor(tensor_kind, ignore=ignore)
    ignore_pattern = rf"(?:\s|{ignore.re.pattern})+" if ignore is not None else r"\s+"

    list_ = Forward()

    list_ <<= Regex(
        rf"\((?:{ignore_pattern})?(?:{tensor.re.pattern}{ignore_pattern})*{tensor.re.pattern}(?:{ignore_pattern})?\)"
    ).add_parse_action(
        lambda tks: [_parse_ascii_field(tks[0], tensor_kind, ignore=ignore)]
    )

    def count_parse_action(tks: ParseResults) -> None:
        nonlocal list_
        length = tks[0]
        assert isinstance(length, int)

        list_ <<= (
            Regex(
                rf"\((?:{ignore_pattern})?(?:{tensor.re.pattern}{ignore_pattern}){{{length - 1}}}{tensor.re.pattern}(?:{ignore_pattern})?\)"
            ).add_parse_action(
                lambda tks: [_parse_ascii_field(tks[0], tensor_kind, ignore=ignore)]
            )
            | Regex(
                rf"\((?s:.{{{length * tensor_kind.size * 8}}}|.{{{length * tensor_kind.size * 4}}})\)"
            ).add_parse_action(
                lambda tks: [
                    _unpack_binary_field(
                        tks[0][1:-1].encode("latin-1"), tensor_kind, length=length
                    )
                ]
            )
            | (
                Literal("{").suppress() + tensor + Literal("}").suppress()
            ).add_parse_action(
                lambda tks: [np.full((length, *tensor_kind.shape), tks[0], dtype=float)]
            )
        )

    count = common.integer.copy().add_parse_action(count_parse_action)

    return (
        Opt(Literal("List") + Literal("<") + str(tensor_kind) + Literal(">")).suppress()
        + Opt(count).suppress()
        + list_
    )


def _dict_of(
    keyword: ParserElement, data: ParserElement, *, located: bool = False
) -> ParserElement:
    dict_ = Forward()

    keyword_entry = keyword + (dict_ | (data + Literal(";").suppress()))

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
    located: bool = False,
) -> ParserElement:
    keyword_entry = keyword + (
        _dict_of(keyword, data, located=located) | (data + Literal(";").suppress())
    )

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
_TENSOR = (
    _tensor(TensorKind.SCALAR)
    | _tensor(TensorKind.VECTOR)
    | _tensor(TensorKind.SYMM_TENSOR)
    | _tensor(TensorKind.TENSOR)
)
_IDENTIFIER = Forward()
_IDENTIFIER <<= Combine(
    Word(_IDENTCHARS, _IDENTBODYCHARS, exclude_chars="()")
    + Opt(Literal("(") + _IDENTIFIER + Literal(")"))
)
_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (Keyword("uniform", _IDENTBODYCHARS).suppress() + _TENSOR) | (
    Keyword("nonuniform", _IDENTBODYCHARS).suppress()
    + (
        _tensor_list(TensorKind.SCALAR, ignore=_COMMENT)
        | _tensor_list(TensorKind.VECTOR, ignore=_COMMENT)
        | _tensor_list(TensorKind.SYMM_TENSOR, ignore=_COMMENT)
        | _tensor_list(TensorKind.TENSOR, ignore=_COMMENT)
    )
)
_TOKEN = dbl_quoted_string | _IDENTIFIER
_DATA = Forward()
_KEYWORD_ENTRY = _keyword_entry_of(_TOKEN | _list_of(_IDENTIFIER), _DATA)
_DICT = _dict_of(_TOKEN, _DATA)
_DATA_ENTRY = Forward()
_LIST_ENTRY = _DICT | _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = _list_of(_LIST_ENTRY)
_NUMBER = common.signed_integer ^ common.ieee_float
_DATA_ENTRY <<= _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | _NUMBER | _SWITCH | _TOKEN

_DATA <<= (
    _DATA_ENTRY[1, ...]
    .set_parse_action(lambda tks: [tuple(tks)] if len(tks) > 1 else [tks[0]])
    .ignore(_COMMENT)
    .parse_with_tabs()
)


def parse_data(s: str) -> Data:
    return cast(Data, _DATA.parse_string(s, parse_all=True)[0])


_LOCATED_DICTIONARY = Group(
    _keyword_entry_of(_TOKEN, Opt(_DATA, default=""), located=True)
)[...]
_LOCATED_DATA = Group(Located(_DATA.copy().add_parse_action(lambda tks: ["", tks[0]])))

_FILE = (
    Dict(_LOCATED_DICTIONARY + Opt(_LOCATED_DATA) + _LOCATED_DICTIONARY)
    .ignore(_COMMENT)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore [no-untyped-call]
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
                r = cast(File, v)

            assert isinstance(r, dict)
            if keywords:
                r[keywords[-1]] = {} if data is ... else data
            else:
                assert data is not ...
                r[None] = data

        return ret
