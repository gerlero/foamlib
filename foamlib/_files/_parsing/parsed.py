from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, cast

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Iterator, Sequence
else:
    from typing import Collection, Iterator, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(...)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping, with_default
from pyparsing import ParseException, ParseResults

from .._util import add_to_mapping
from ._grammar import FILE

if TYPE_CHECKING:
    from .._typing import Data, File, StandaloneData, SubDict


class Parsed(
    MutableMultiMapping["tuple[str, ...]", "Data | StandaloneData | EllipsisType"]
):
    @dataclasses.dataclass
    class _Entry:
        data: Data | StandaloneData | EllipsisType
        start: int
        end: int

    def __init__(self, contents: bytes | str) -> None:
        if isinstance(contents, bytes):
            contents_str = contents.decode("latin-1")
        else:
            contents_str = contents
            contents = contents.encode("latin-1")

        try:
            parse_results = FILE.parse_string(contents_str, parse_all=True)
        except ParseException as e:
            msg = f"Failed to parse contents: {e}"
            raise ValueError(msg) from e
        self._parsed = self._flatten_results(parse_results)

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_results(
        parse_results: ParseResults | Sequence[ParseResults],
        *,
        _keywords: tuple[str, ...] = (),
    ) -> MultiDict[tuple[str, ...], Parsed._Entry]:
        ret: MultiDict[tuple[str, ...], Parsed._Entry] = MultiDict()
        for parse_result in parse_results:
            value = parse_result.value
            assert isinstance(value, Sequence)
            start = parse_result.locn_start
            assert isinstance(start, int)
            end = parse_result.locn_end
            assert isinstance(end, int)
            keyword, *data = value
            if keyword is None:
                assert not _keywords
                assert len(data) == 1
                assert not isinstance(data[0], ParseResults)
                assert () not in ret
                ret[()] = Parsed._Entry(data[0], start, end)
            else:
                assert isinstance(keyword, str)
                if len(data) == 0 or isinstance(data[0], ParseResults):
                    if (*_keywords, keyword) in ret:
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret[(*_keywords, keyword)] = Parsed._Entry(..., start, end)
                    ret.extend(
                        Parsed._flatten_results(data, _keywords=(*_keywords, keyword))
                    )
                else:
                    if (*_keywords, keyword) in ret and not keyword.startswith("#"):
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret.add((*_keywords, keyword), Parsed._Entry(data[0], start, end))
        return ret

    @override
    @with_default
    def getall(
        self, keywords: tuple[str, ...]
    ) -> Collection[Data | StandaloneData | EllipsisType]:
        return [entry.data for entry in self._parsed.getall(keywords)]

    @override
    def __setitem__(
        self, key: tuple[str, ...], value: Data | StandaloneData | EllipsisType
    ) -> None:  # pragma: no cover
        msg = "Use 'put' method instead"
        raise NotImplementedError(msg)

    def put(
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        start, end = self.entry_location(keywords)

        self._update_content(start, end, content)
        self._parsed[keywords] = Parsed._Entry(data, start, start + len(content))
        self._remove_child_entries(keywords)

    @override
    def add(
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType,
        content: bytes,
    ) -> None:
        assert keywords not in self._parsed or (
            keywords and keywords[-1].startswith("#")
        )
        assert not keywords or not keywords[-1].startswith("#") or data is not ...
        assert keywords or data is not ...

        start, end = self.entry_location(keywords, add=True)

        self._parsed.add(keywords, Parsed._Entry(data, start, end))
        self._update_content(start, end, content)

    @override
    @with_default
    def popone(self, keywords: tuple[str, ...]) -> Data | StandaloneData | EllipsisType:
        start, end = self.entry_location(keywords)
        entry = self._parsed.popone(keywords)
        self._remove_child_entries(keywords)
        self._update_content(start, end, b"")
        return entry.data

    @override
    def __contains__(self, keywords: object) -> bool:
        return keywords in self._parsed

    @override
    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self._parsed)

    @override
    def __len__(self) -> int:
        return len(self._parsed)

    def _update_content(self, start: int, end: int, new_content: bytes) -> None:
        """Update content and adjust positions of other entries."""
        diff = len(new_content) - (end - start)

        # Update positions of other entries if content length changed
        if diff != 0:
            for entry in self._parsed.values():
                assert isinstance(entry, Parsed._Entry)
                if entry.start >= end:
                    entry.start += diff
                    entry.end += diff
                elif entry.end > start:
                    entry.end += diff

        self.contents = self.contents[:start] + new_content + self.contents[end:]
        self.modified = True

    def _remove_child_entries(self, keywords: tuple[str, ...]) -> None:
        """Remove all child entries of the given keywords."""
        for k in list(self._parsed):
            if keywords != k and keywords == k[: len(keywords)]:
                del self._parsed[k]

    def entry_location(
        self, keywords: tuple[str, ...], *, add: bool = False
    ) -> tuple[int, int]:
        if add or keywords not in self._parsed:
            if len(keywords) > 1:
                assert self[keywords[:-1]] is ...
                start, end = self.entry_location(keywords[:-1])
                end = self.contents.rindex(b"}", start, end)
            else:
                end = len(self.contents)

            start = end
        else:
            entry = self._parsed[keywords]
            start = entry.start
            end = entry.end

        return start, end

    def as_dict(self) -> File:
        ret: File = {}
        for keywords, entry in self._parsed.items():
            assert isinstance(entry, Parsed._Entry)
            if not keywords:
                assert entry.data is not ...
                assert None not in ret
                ret[None] = entry.data
            elif entry.data is ...:
                parent: File | SubDict = ret
                for k in keywords[:-1]:
                    sub = parent[k]
                    assert isinstance(sub, (dict, MultiDict))
                    parent = sub
                assert keywords[-1] not in parent
                parent[keywords[-1]] = {}
            elif len(keywords) == 1:
                ret = add_to_mapping(ret, keywords[0], entry.data)
            else:
                grandparent: File | SubDict = ret
                for k in keywords[:-2]:
                    sub = grandparent[k]
                    assert isinstance(sub, (dict, MultiDict))
                    grandparent = sub
                sub = grandparent[keywords[-2]]
                assert isinstance(sub, (dict, MultiDict))
                grandparent[keywords[-2]] = add_to_mapping(
                    sub, keywords[-1], cast("Data", entry.data)
                )

        return ret
