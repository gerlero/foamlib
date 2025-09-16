from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from pyparsing import (
    Forward,
    Group,
    LineEnd,
    Literal,
    Located,
    NoMatch,
    Opt,
    ParserElement,
    ParseResults,
    Regex,
    common,
    counted_array,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

from .._util import as_dict_check_unique


def ascii_numeric_list(
    dtype: DTypeLike,
    *,
    nested: int | None = None,
    ignore: Regex | None = None,
    empty_ok: bool = False,
) -> ParserElement:
    dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.floating):
        element = common.ieee_float
        element_pattern = r"(?i:[+-]?(?:(?:\d+\.?\d*(?:e[+-]?\d+)?)|nan|inf(?:inity)?))"
    elif np.issubdtype(dtype, np.integer):
        element = common.integer
        element_pattern = r"(?:-?\d+)"
    else:
        msg = f"Unsupported dtype: {dtype}"
        raise TypeError(msg)

    spacing_pattern = (
        rf"(?:(?:\s|{ignore.re.pattern})+)" if ignore is not None else r"(?:\s+)"
    )

    if nested is not None:
        element = (
            Literal("(").suppress() + Group(element[nested]) + Literal(")").suppress()
        )
        element_pattern = rf"(?:{spacing_pattern}?\({element_pattern}?(?:{element_pattern}{spacing_pattern}){{{nested - 1}}}{element_pattern}{spacing_pattern}?\))"

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_

        if not tks:
            count = None
        else:
            (count,) = tks
            assert isinstance(count, int)

        if count is None:
            if not empty_ok:
                list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}{spacing_pattern}?\)"
            else:
                list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}?{spacing_pattern}?\)"

        elif count == 0:
            if not empty_ok:
                list_ <<= NoMatch()
            else:
                list_ <<= (Literal("(") + Literal(")")).add_parse_action(
                    lambda: np.empty((0, nested) if nested else 0, dtype=dtype)
                )
            return

        else:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern}){{{count - 1}}}{element_pattern}{spacing_pattern}?\)"

        list_ <<= Regex(list_pattern).add_parse_action(
            lambda tks: to_array(tks, dtype=dtype)
        )

    def to_array(
        tks: ParseResults, *, dtype: DTypeLike
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        (s,) = tks
        assert s.startswith("(")
        assert s.endswith(")")
        s = s[1:-1]
        if ignore is not None:
            s = re.sub(ignore.re, " ", s)
        if nested is not None:
            s = s.replace("(", " ").replace(")", " ")

        ret: np.ndarray[
            tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]
        ] = np.fromstring(s, sep=" ", dtype=dtype)

        if nested is not None:
            ret = ret.reshape(-1, nested)

        return ret

    def to_full_array(
        tks: ParseResults, *, dtype: type
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.integer | np.floating]]:
        count, lst = tks
        assert isinstance(count, int)

        if nested is None:
            return np.full(count, lst, dtype=dtype)

        return np.full((count, nested), lst, dtype=dtype)  # type: ignore[return-value]

    ret = ((Opt(common.integer).add_parse_action(process_count)).suppress() + list_) | (
        common.integer + Literal("{").suppress() + element + Literal("}").suppress()
    ).add_parse_action(lambda tks: to_full_array(tks, dtype=float))

    if ignore is not None:
        ret.ignore(ignore)

    return ret


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


def ascii_face_list(*, ignore: Regex | None = None) -> ParserElement:
    element_pattern = r"(?:-?\d+)"
    spacing_pattern = (
        rf"(?:(?:\s|{ignore.re.pattern})+)" if ignore is not None else r"(?:\s+)"
    )

    element_pattern = rf"(?:(?:3{spacing_pattern}?\((?:{element_pattern}{spacing_pattern}){{2}}{element_pattern}{spacing_pattern}?\))|(?:4{spacing_pattern}?\((?:{element_pattern}{spacing_pattern}){{3}}{element_pattern}{spacing_pattern}?\)))"

    list_ = Forward()

    def process_count(tks: ParseResults) -> None:
        nonlocal list_
        if not tks:
            count = None
        else:
            (count,) = tks
            assert isinstance(count, int)

        if count is None:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern})*{element_pattern}{spacing_pattern}?\)"

        elif count == 0:
            list_ <<= NoMatch()
            return

        else:
            list_pattern = rf"\({spacing_pattern}?(?:{element_pattern}{spacing_pattern}){{{count - 1}}}{element_pattern}{spacing_pattern}?\)"

        list_ <<= Regex(list_pattern).add_parse_action(to_face_list)

    def to_face_list(
        tks: ParseResults,
    ) -> list[list[np.ndarray[tuple[int], np.dtype[np.int64]]]]:
        (s,) = tks
        assert s.startswith("(")
        assert s.endswith(")")
        if ignore is not None:
            s = re.sub(ignore.re, " ", s)
        s = s.replace("(", " ").replace(")", " ")

        raw = np.fromstring(s, sep=" ", dtype=int)

        values: list[np.ndarray[tuple[int], np.dtype[np.int64]]] = []
        i = 0
        while i < raw.size:
            assert raw[i] in (3, 4)
            values.append(raw[i + 1 : i + raw[i] + 1])
            i += raw[i] + 1

        return [values]

    return Opt(common.integer).add_parse_action(process_count).suppress() + list_


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
