from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import numpy as np
from pyparsing import (
    Forward,
    Group,
    LineEnd,
    Literal,
    Located,
    NoMatch,
    ParseException,
    ParserElement,
    ParseResults,
    Regex,
    common,
    counted_array,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

from .._util import as_dict_check_unique


class ASCIINumericList(ParserElement):
    _FLOAT_PATTERN = r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))"
    _INT_PATTERN = r"-?\d+"

    def __init__(
        self,
        dtype: DTypeLike,
        elshape: tuple[()] | tuple[int] = (),
        *,
        empty_ok: bool = False,
    ) -> None:
        super().__init__()
        self._dtype = np.dtype(dtype)
        self._elshape = elshape
        self._empty_ok = empty_ok

        self.name = f"ASCIINumericList({self._dtype})"

    @override
    def _generateDefaultName(self) -> str:
        return self.name

    @override
    def parseImpl(
        self, instring: str, loc: int, doActions: bool = True
    ) -> tuple[int, ParseResults]:
        ignore_pattern = (
            "|".join(
                rf"(?:{ignore_expr.expr.re.pattern})"  # type: ignore [attr-defined]
                for ignore_expr in ignore_exprs
            )
            if (ignore_exprs := self.ignoreExprs)
            else ""
        )
        spacing_pattern = rf"\s|(?:{ignore_pattern})" if ignore_pattern else r"\s"

        if np.issubdtype(self._dtype, np.floating):
            base_pattern = self._FLOAT_PATTERN
        elif np.issubdtype(self._dtype, np.integer):
            base_pattern = self._INT_PATTERN
        else:
            msg = f"Unsupported dtype {self._dtype}"
            raise TypeError(msg)

        if self._elshape:
            (dim,) = self._elshape
            element_pattern = rf"\((?:{spacing_pattern})*(?:{base_pattern}(?:{spacing_pattern})*){{{dim}}}\)"
        else:
            element_pattern = base_pattern

        regular_pattern = re.compile(
            rf"(\d*)(?:{spacing_pattern})*\((?:{spacing_pattern})*((?:{element_pattern}(?:{spacing_pattern})*)*)\)"
        )
        assert regular_pattern.groups == 2

        if match := regular_pattern.match(instring, pos=loc):
            count = int(c) if (c := match.group(1)) else None
            contents = match.group(2)
            if ignore_pattern:
                contents = re.sub(ignore_pattern, " ", contents)
            if self._elshape:
                contents = contents.replace("(", " ").replace(")", " ")

            arr = np.fromstring(contents, dtype=self._dtype, sep=" ")

            if self._elshape:
                arr = arr.reshape(-1, *self._elshape)

            if count is not None and arr.shape[0] != count:
                msg = f"Expected {count} elements, got {arr.shape[0]}"
                raise ParseException(
                    instring,
                    loc,
                    msg,
                    self,
                )

            if arr.size == 0 and not self._empty_ok:
                msg = "Expected at least one element"
                raise ParseException(
                    instring,
                    loc,
                    msg,
                    self,
                )

            return match.end(), ParseResults(arr)

        repeated_pattern = re.compile(
            rf"(\d+)(?:{spacing_pattern})*{{(?:{spacing_pattern})*({element_pattern})(?:{spacing_pattern})*\}}"
        )
        assert repeated_pattern.groups == 2

        if match := repeated_pattern.match(instring, pos=loc):
            count = int(match.group(1))
            contents = match.group(2)
            if ignore_pattern:
                contents = re.sub(ignore_pattern, " ", contents)
            if self._elshape:
                contents = contents.replace("(", " ").replace(")", " ")

            arr = np.fromstring(contents, dtype=self._dtype, sep=" ")

            if self._elshape:
                arr = arr.reshape(-1, *self._elshape)

            arr = np.repeat(arr, count, axis=0)

            return match.end(), ParseResults(arr)

        msg = "Expected ASCII numeric list"
        raise ParseException(
            instring,
            loc,
            msg,
            self,
        )


def binary_numeric_list(
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


class ASCIIFacesLikeList(ParserElement):
    _INT_PATTERN = ASCIINumericList._INT_PATTERN

    def __init__(self) -> None:
        super().__init__()
        self.name = "ASCIIFacesLikeList"

    @override
    def _generateDefaultName(self) -> str:
        return self.name

    @override
    def parseImpl(
        self, instring: str, loc: int, doActions: bool = True
    ) -> tuple[int, ParseResults]:
        ignore_pattern = (
            "|".join(
                rf"(?:{ignore_expr.expr.re.pattern})"  # type: ignore [attr-defined]
                for ignore_expr in ignore_exprs
            )
            if (ignore_exprs := self.ignoreExprs)
            else ""
        )
        spacing_pattern = rf"\s|(?:{ignore_pattern})" if ignore_pattern else r"\s"

        three_face_pattern = rf"3(?:{spacing_pattern})*\((?:{spacing_pattern})*(?:{self._INT_PATTERN}(?:{spacing_pattern})*){{3}}\)"
        four_face_pattern = rf"4(?:{spacing_pattern})*\((?:{spacing_pattern})*(?:{self._INT_PATTERN}(?:{spacing_pattern})*){{4}}\)"

        face_pattern = rf"(?:{three_face_pattern})|(?:{four_face_pattern})"

        face_list_pattern = re.compile(
            rf"(\d*)(?:{spacing_pattern})*\((?:{spacing_pattern})*((?:(?:{face_pattern})(?:{spacing_pattern})*)+)\)"
        )
        assert face_list_pattern.groups == 2

        if match := face_list_pattern.match(instring, pos=loc):
            count = int(c) if (c := match.group(1)) else None
            contents = match.group(2)
            if ignore_pattern:
                contents = re.sub(ignore_pattern, " ", contents)
            contents = contents.replace("(", " ").replace(")", " ")

            raw = np.fromstring(contents, sep=" ", dtype=int)
            assert raw.size > 0

            values: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
            i = 0
            while i < raw.size:
                assert raw[i] in (3, 4)
                values.append(raw[i + 1 : i + raw[i] + 1])
                i += raw[i] + 1

            if count is not None and len(values) != count:
                msg = f"Expected {count} elements, got {len(values)}"
                raise ParseException(
                    instring,
                    loc,
                    msg,
                    self,
                )

            return match.end(), ParseResults([values])

        msg = "Expected ASCII faces-like list"
        raise ParseException(
            instring,
            loc,
            msg,
            self,
        )


def list_of(entry: ParserElement) -> ParserElement:
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


def dict_of(
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


def keyword_entry_of(
    keyword: ParserElement,
    data: ParserElement,
    *,
    directive: ParserElement | None = None,
    data_entry: ParserElement | None = None,
    located: bool = False,
) -> ParserElement:
    keyword_entry = keyword + (
        dict_of(
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
