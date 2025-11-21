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
    ParseExpression,
    ParserElement,
    ParseResults,
    Regex,
    Suppress,
    common,
    counted_array,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    if sys.version_info >= (3, 9):
        from collections.abc import Iterable
    else:
        from typing import Iterable

    from numpy.typing import DTypeLike

from .._util import as_dict_check_unique


class MatchLongest(ParseExpression):
    @override
    def __init__(self, exprs: Iterable[ParserElement], savelist: bool = False) -> None:
        super().__init__(exprs, savelist)

    @override
    def parseImpl(
        self, instring: str, loc: int, doActions: bool = True
    ) -> tuple[int, ParseResults]:
        best_loc = -1
        best_tks: ParseResults | None = None

        for expr in self.exprs:
            try:
                next_loc, tks = expr._parse(instring, loc, doActions)  # ty: ignore[call-non-callable]
            except ParseException:
                continue

            if next_loc > best_loc:
                best_loc = next_loc
                best_tks = tks

        if best_loc >= 0:
            assert isinstance(best_tks, ParseResults)
            return best_loc, best_tks

        msg = "No alternatives matched"
        raise ParseException(
            instring,
            loc,
            msg,
            self,
        )


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
        spacing_pattern = "|".join(re.escape(c) for c in self.whiteChars)
        assert spacing_pattern

        assert all(
            isinstance(ignore_expr, Suppress) and isinstance(ignore_expr.expr, Regex)
            for ignore_expr in self.ignoreExprs
        )
        ignore_pattern = "|".join(
            ignore_expr.expr.re.pattern  # ty: ignore[unresolved-attribute]
            for ignore_expr in self.ignoreExprs
        )

        if ignore_pattern:
            spacing_pattern = f"{spacing_pattern}|{ignore_pattern}"

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

            if not all(c.isspace() for c in self.whiteChars):
                contents = re.sub(spacing_pattern, " ", contents)
            elif ignore_pattern:
                contents = re.sub(ignore_pattern, " ", contents)  # Faster

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
            contents = re.sub(spacing_pattern, " ", contents)
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
    dtype: DTypeLike, elshape: tuple[()] | tuple[int] = (), *, empty_ok: bool = False
) -> ParserElement:
    dtype = np.dtype(dtype)

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_
        (count,) = tks
        assert isinstance(count, int)

        if count == 0 and not empty_ok:
            list_ <<= NoMatch()
            return

        if not elshape:
            elsize = 1
        else:
            (elsize,) = elshape

        list_ <<= Regex(rf"\((?s:{'.' * dtype.itemsize * elsize}){{{count}}}\)")

    def to_array(
        tks: ParseResults,
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        count, contents = tks
        assert isinstance(count, int)
        assert isinstance(contents, str)
        assert contents[0] == "("
        assert contents[-1] == ")"
        contents = contents[1:-1]

        ret = np.frombuffer(contents.encode("latin-1"), dtype=dtype)

        if elshape:
            ret = ret.reshape(count, *elshape)

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
        spacing_pattern = "|".join(re.escape(c) for c in self.whiteChars)
        assert spacing_pattern

        assert all(
            isinstance(ignore_expr, Suppress) and isinstance(ignore_expr.expr, Regex)
            for ignore_expr in self.ignoreExprs
        )
        ignore_pattern = "|".join(
            ignore_expr.expr.re.pattern  # ty: ignore[unresolved-attribute]
            for ignore_expr in self.ignoreExprs
        )

        if ignore_pattern:
            spacing_pattern = f"{spacing_pattern}|{ignore_pattern}"

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

            if not all(c.isspace() for c in self.whiteChars):
                contents = re.sub(spacing_pattern, " ", contents)
            elif ignore_pattern:
                contents = re.sub(ignore_pattern, " ", contents)  # Faster

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
