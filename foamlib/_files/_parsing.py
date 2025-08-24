from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Tuple, Union, cast, overload

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(...)

import numpy as np
from multidict import MultiDict
from pyparsing import (
    CaselessKeyword,
    CharsNotIn,
    Combine,
    Dict,
    Forward,
    Group,
    Keyword,
    LineEnd,
    Literal,
    Located,
    NoMatch,
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

from ._types import Data, Dimensioned, DimensionSet, File, StandaloneData, SubDict

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _ascii_numeric_list(
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


def _binary_numeric_list(
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


def _ascii_face_list(*, ignore: Regex | None = None) -> ParserElement:
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


def _list_of(entry: ParserElement) -> ParserElement:
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


def _dict_of(
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
        Literal("{").suppress()
        + Dict(Group(keyword_entry)[...], asdict=not located)
        + Literal("}").suppress()
    )

    return dict_


def _keyword_entry_of(
    keyword: ParserElement,
    data: ParserElement,
    *,
    directive: ParserElement | None = None,
    data_entry: ParserElement | None = None,
    located: bool = False,
) -> ParserElement:
    keyword_entry = keyword + (
        _dict_of(
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
).add_parse_action(lambda tks: np.array(tks[0], dtype=float))
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
                _ascii_numeric_list(dtype=float, ignore=_COMMENT, empty_ok=True)
                | _binary_numeric_list(dtype=np.float64, empty_ok=True)
                | _binary_numeric_list(dtype=np.float32, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("vector") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=3, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=3, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=3, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("symmTensor") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=6, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=6, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=6, empty_ok=True)
            )
        )
        | (
            Opt(
                Literal("List") + Literal("<") + Literal("tensor") + Literal(">")
            ).suppress()
            + (
                _ascii_numeric_list(
                    dtype=float, nested=9, ignore=_COMMENT, empty_ok=True
                )
                | _binary_numeric_list(np.float64, nested=9, empty_ok=True)
                | _binary_numeric_list(np.float32, nested=9, empty_ok=True)
            )
        )
    )
)

_DIRECTIVE = Word("#", _IDENTBODYCHARS)
_TOKEN = dbl_quoted_string | _DIRECTIVE | _IDENTIFIER
_DATA = Forward()
_DATA_ENTRY = Forward()
_KEYWORD_ENTRY = _keyword_entry_of(
    _TOKEN | _list_of(_IDENTIFIER),
    Opt(_DATA, default=""),
    directive=_DIRECTIVE,
    data_entry=_DATA_ENTRY,
)
_DICT = _dict_of(_TOKEN, Opt(_DATA, default=""))
_LIST_ENTRY = _DICT | _KEYWORD_ENTRY | _DATA_ENTRY
_LIST = _list_of(_LIST_ENTRY)
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
    _ascii_numeric_list(dtype=int, ignore=_COMMENT)
    | _ascii_face_list(ignore=_COMMENT)
    | _ascii_numeric_list(dtype=float, nested=3, ignore=_COMMENT)
    | (
        _binary_numeric_list(dtype=np.int64) + Opt(_binary_numeric_list(dtype=np.int64))
    ).add_parse_action(lambda tks: tuple(tks) if len(tks) > 1 else tks[0])
    | (
        _binary_numeric_list(dtype=np.int32) + Opt(_binary_numeric_list(dtype=np.int32))
    ).add_parse_action(lambda tks: tuple(tks) if len(tks) > 1 else tks[0])
    | _binary_numeric_list(dtype=np.float64, nested=3)
    | _binary_numeric_list(dtype=np.float32, nested=3)
    | _DATA
).add_parse_action(lambda tks: [None, tks[0]])


_FILE = (
    Dict(_KEYWORD_ENTRY[...] + Opt(Group(_STANDALONE_DATA)) + _KEYWORD_ENTRY[...])
    .ignore(_COMMENT)
    .parse_with_tabs()
)

_DATA_OR_DICT = (_DATA | _DICT).ignore(_COMMENT).parse_with_tabs()


@overload
def loads(s: bytes | str, *, keywords: tuple[()]) -> Union[File, MultiDict, StandaloneData]: ...


@overload
def loads(
    s: bytes | str, *, keywords: tuple[str, ...] | None = None
) -> Union[File, MultiDict, StandaloneData, Data, SubDict]: ...


def loads(
    s: bytes | str, *, keywords: tuple[str, ...] | None = None
) -> Union[File, MultiDict, StandaloneData, Data, SubDict]:
    if isinstance(s, bytes):
        s = s.decode("latin-1")

    if keywords == ():
        # Try the original approach first
        try:
            data = _FILE.parse_string(s, parse_all=True).as_dict()
        except Exception:
            # If that fails, fall back to Parsed
            parsed = Parsed(s.encode("latin-1"))
            data = parsed.as_dict()
        else:
            # Check if we might have lost duplicates by parsing with Parsed too
            parsed = Parsed(s.encode("latin-1"))
            parsed_data = parsed.as_dict()
            
            # If the Parsed version has MultiDict but original doesn't, use Parsed version
            if isinstance(parsed_data, dict):
                for v in parsed_data.values():
                    if isinstance(v, MultiDict):
                        data = parsed_data
                        break

        if len(data) == 1 and None in data:
            data = data[None]

    else:
        data = _DATA_OR_DICT.parse_string(s, parse_all=True)[0]

    return data


_LOCATED_KEYWORD_ENTRIES = Group(
    _keyword_entry_of(
        _TOKEN,
        Opt(_DATA, default=""),
        directive=_DIRECTIVE,
        data_entry=_DATA_ENTRY,
        located=True,
    )
)[...]
_LOCATED_STANDALONE_DATA = Group(Located(_STANDALONE_DATA))

_LOCATED_FILE = (
    Dict(
        _LOCATED_KEYWORD_ENTRIES
        + Opt(_LOCATED_STANDALONE_DATA)
        + _LOCATED_KEYWORD_ENTRIES
    )
    .ignore(_COMMENT)
    .parse_with_tabs()
)


class Parsed(Mapping[Tuple[str, ...], Union[Data, StandaloneData, EllipsisType]]):
    """Parser for OpenFOAM data files with support for multiple directives.
    
    This class parses OpenFOAM files and preserves multiple #-directives with
    the same keyword (e.g., multiple #includeFunc entries), maintaining their
    order and relative positions within the file.
    """
    def __init__(self, contents: bytes) -> None:
        # Store all key-value pairs in order, allowing duplicates
        self._items: list[tuple[tuple[str, ...], tuple[int, Data | StandaloneData | EllipsisType, int]]] = []
        
        for parse_result in _LOCATED_FILE.parse_string(
            contents.decode("latin-1"), parse_all=True
        ):
            # Get all key-value pairs from this parse result  
            for key, value in self._flatten_result(parse_result):
                self._items.append((key, value))
        
        # Build a lookup for unique keys
        self._unique_keys: dict[tuple[str, ...], tuple[int, Data | StandaloneData | EllipsisType, int]] = {}
        for key, value in self._items:
            if key not in self._unique_keys:
                self._unique_keys[key] = value

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_result(
        parse_result: ParseResults, *, _keywords: tuple[str, ...] = ()
    ) -> Sequence[
        tuple[tuple[str, ...], tuple[int, Data | StandaloneData | EllipsisType, int]]
    ]:
        # Return a sequence of (key, value) pairs to preserve order and allow duplicates
        ret: list[
            tuple[tuple[str, ...], tuple[int, Data | StandaloneData | EllipsisType, int]]
        ] = []
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
            ret.append(((), (start, data[0], end)))
        else:
            assert isinstance(keyword, str)
            # First add the placeholder entry
            ret.append(((*_keywords, keyword), (start, ..., end)))
            for d in data:
                if isinstance(d, ParseResults):
                    # Recursively process nested results
                    nested_results = Parsed._flatten_result(d, _keywords=(*_keywords, keyword))
                    ret.extend(nested_results)
                else:
                    # Add each data item as a separate entry - this preserves multiple directives
                    ret.append(((*_keywords, keyword), (start, d, end)))
        return ret

    def __getitem__(
        self, keywords: tuple[str, ...]
    ) -> Data | StandaloneData | EllipsisType:
        """Get the first value for the given keywords (backward compatibility)."""
        if keywords not in self._unique_keys:
            raise KeyError(keywords)
        
        # For backward compatibility, return the first non-ellipsis value if available
        for key, (_, data, _) in self._items:
            if key == keywords:
                if data is not ...:
                    return data
        
        # If no non-ellipsis values found, return the first value (could be ellipsis)
        _, data, _ = self._unique_keys[keywords]
        return data

    def getall(
        self, keywords: tuple[str, ...]
    ) -> list[Data | StandaloneData | EllipsisType]:
        """Get all values for the given keywords, preserving order.
        
        This method is particularly useful for accessing multiple #-directives
        with the same keyword (e.g., multiple #includeFunc entries), which 
        were previously overwritten by the parsing system.
        
        Args:
            keywords: Tuple of keywords identifying the entry
            
        Returns:
            List of all values for the keywords, preserving their order of
            appearance in the original file
            
        Raises:
            KeyError: If the keywords are not found
            
        Example:
            >>> p = Parsed(b'''
            ... functions
            ... {
            ...     #includeFunc first
            ...     #includeFunc second
            ...     #includeFunc third
            ... }
            ... ''')
            >>> p.getall(('functions', '#includeFunc'))
            ['first', 'second', 'third']
            >>> p[('functions', '#includeFunc')]  # Single access returns first
            'first'
        """
        if keywords not in self._unique_keys:
            raise KeyError(keywords)
        
        # Get all values for this key in order
        values = []
        for key, (_, data, _) in self._items:
            if key == keywords and data is not ...:  # Skip placeholder entries
                values.append(data)
        return values

    def put(
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        start, end = self.entry_location(keywords, missing_ok=True)

        diff = len(content) - (end - start)
        
        # Update all items positions
        updated_items = []
        for key, (s, d, e) in self._items:
            if s >= end:
                updated_items.append((key, (s + diff, d, e + diff)))
            elif e > start:
                updated_items.append((key, (s, d, e + diff)))
            else:
                updated_items.append((key, (s, d, e)))
        
        # Add the new entry
        updated_items.append((keywords, (start, data, end + diff)))
        self._items = updated_items
        
        # Update unique keys lookup
        self._unique_keys[keywords] = (start, data, end + diff)

        self.contents = self.contents[:start] + content + self.contents[end:]
        self.modified = True

        # Remove nested entries
        keys_to_remove = []
        for key, _ in self._items:
            if keywords != key and keywords == key[: len(keywords)]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._items = [(k, v) for k, v in self._items if k != key]
            if key in self._unique_keys:
                del self._unique_keys[key]

    def __delitem__(self, keywords: tuple[str, ...]) -> None:
        start, end = self.entry_location(keywords)
        
        # Remove from unique keys
        del self._unique_keys[keywords]
        
        # Remove all occurrences from items
        self._items = [(k, v) for k, v in self._items if k != keywords]

        # Remove nested entries
        keys_to_remove = []
        for key, _ in self._items:
            if keywords == key[: len(keywords)]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._items = [(k, v) for k, v in self._items if k != key]
            if key in self._unique_keys:
                del self._unique_keys[key]

        diff = end - start
        # Update all items positions
        updated_items = []
        for key, (s, d, e) in self._items:
            if s > end:
                updated_items.append((key, (s - diff, d, e - diff)))
            elif e > start:
                updated_items.append((key, (s, d, e - diff)))
            else:
                updated_items.append((key, (s, d, e)))
        self._items = updated_items
        
        # Update unique keys lookup
        updated_unique = {}
        for key, (s, d, e) in self._items:
            if key not in updated_unique:
                updated_unique[key] = (s, d, e)
        self._unique_keys = updated_unique

        self.contents = self.contents[:start] + self.contents[end:]
        self.modified = True

    def __contains__(self, keywords: object) -> bool:
        return keywords in self._unique_keys

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self._unique_keys)

    def __len__(self) -> int:
        return len(self._unique_keys)

    def entry_location(
        self, keywords: tuple[str, ...], *, missing_ok: bool = False
    ) -> tuple[int, int]:
        try:
            start, _, end = self._unique_keys[keywords]
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

    def as_dict(self) -> Union[File, MultiDict]:
        """Return a nested dict representation, using MultiDict if there are duplicates."""
        # First, check if we have any duplicate keys (ignoring ellipsis placeholders)
        key_counts = {}
        for key, (_, data, _) in self._items:
            if data is not ...:  # Only count actual data, not placeholders
                key_counts[key] = key_counts.get(key, 0) + 1
        
        has_duplicates = any(count > 1 for count in key_counts.values())
        
        if not has_duplicates:
            # Use regular dict approach when no duplicates
            ret: File = {}
            
            # First collect all actual data values (non-ellipsis)
            actual_data = {}
            for keywords, (_, data, _) in self._items:
                if data is not ... and keywords not in actual_data:
                    actual_data[keywords] = data
            
            # Identify all subdictories that need to be created
            subdirs_needed = set()
            for keywords in actual_data:
                # Add all parent paths that need subdirectories
                for i in range(1, len(keywords)):
                    parent_path = keywords[:i]
                    subdirs_needed.add(parent_path)
            
            # Create subdirectories first  
            for subdir_path in sorted(subdirs_needed):
                r = ret
                for k in subdir_path[:-1]:
                    if k not in r:
                        r[k] = {}
                    r = r[k]
                if subdir_path[-1] not in r:
                    r[subdir_path[-1]] = {}
            
            # Then add actual data
            for keywords, data in actual_data.items():
                r = ret
                if keywords:  # Only navigate if there are keywords
                    for k in keywords[:-1]:
                        assert k in r and isinstance(r[k], dict)
                        r = r[k]
                    r[keywords[-1]] = data
                else:
                    # Handle standalone data (no keywords)
                    r[None] = data
            
            return ret
        else:
            # Build structure that can handle duplicates at any level
            # Group items by their parent path (only for actual data, not placeholders)
            groups: dict[tuple[str, ...], list[tuple[str, Data | StandaloneData | EllipsisType]]] = {}
            
            for keywords, (_, data, _) in self._items:
                if data is ...:  # Skip placeholder entries
                    continue
                    
                if keywords:
                    parent_path = keywords[:-1] 
                    final_key = keywords[-1]
                    if parent_path not in groups:
                        groups[parent_path] = []
                    groups[parent_path].append((final_key, data))
                else:
                    # Top-level entry
                    if () not in groups:
                        groups[()] = []
                    groups[()].append((None, data))
            
            # Also need to handle subdictionaries (ellipsis entries)
            subdicts = set()
            for keywords, (_, data, _) in self._unique_keys.items():
                if data is ... and keywords:
                    subdicts.add(keywords)
            
            # Build the result structure
            ret = {}
            
            # First create subdirectory structure
            for subdict_path in subdicts:
                current = ret
                for k in subdict_path[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                if subdict_path[-1] not in current:
                    current[subdict_path[-1]] = {}
            
            for parent_path, items in groups.items():
                # Navigate to the parent location
                current = ret
                for k in parent_path:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Check if this level has duplicates
                key_counts_at_level = {}
                for key, _ in items:
                    key_counts_at_level[key] = key_counts_at_level.get(key, 0) + 1
                
                level_has_duplicates = any(count > 1 for count in key_counts_at_level.values())
                
                if level_has_duplicates:
                    # Use MultiDict for this level
                    level_dict = MultiDict()
                    for key, data in items:
                        level_dict.add(key, data)
                    
                    # If current is a regular dict, convert it
                    if isinstance(current, dict):
                        # Save existing content
                        existing_items = list(current.items())
                        current.clear()
                        
                        # Add existing items to MultiDict if they don't conflict
                        for existing_key, existing_value in existing_items:
                            if existing_key not in [k for k, _ in items]:
                                level_dict[existing_key] = existing_value
                        
                        # Update the parent to point to the MultiDict
                        if parent_path:
                            parent = ret
                            for k in parent_path[:-1]:
                                parent = parent[k]
                            parent[parent_path[-1]] = level_dict
                        else:
                            ret = level_dict
                    else:
                        # Current is already a MultiDict
                        for key, data in items:
                            current.add(key, data)
                else:
                    # Use regular dict for this level
                    for key, data in items:
                        current[key] = data
            
            return ret
