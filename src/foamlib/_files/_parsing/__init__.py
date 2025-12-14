import sys
from collections.abc import Collection, Iterator
from typing import cast, overload

if sys.version_info >= (3, 11):
    from typing import Never, Unpack
else:
    from typing_extensions import Never, Unpack

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from types import EllipsisType

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping, with_default

from ...typing import Data, FileDict, StandaloneData, SubDict
from .._util import add_to_mapping
from ._parser import (
    ParsedEntry,
    parse,
    parse_located,
)
from .exceptions import FoamFileDecodeError

__all__ = [
    "FoamFileDecodeError",
    "ParsedFile",
    "parse",
]


class ParsedFile(
    MutableMultiMapping[tuple[str, ...], Data | StandaloneData | EllipsisType | None]
):
    def __init__(self, contents: bytearray | bytes, /) -> None:
        if isinstance(contents, bytes):
            contents = bytearray(contents)

        self._parsed = parse_located(contents)
        self.contents = contents
        self.modified = False

    @overload
    @with_default
    def getall(self, keywords: tuple[()], /) -> Collection[StandaloneData]: ...

    @overload
    @with_default
    def getall(
        self, keywords: tuple[str, Unpack[tuple[str, ...]]], /
    ) -> Collection[Data | EllipsisType | None]: ...

    @override
    @with_default
    def getall(
        self, keywords: tuple[str, ...], /
    ) -> Collection[Data | StandaloneData | EllipsisType | None]:
        return [entry.data for entry in self._parsed.getall(keywords)]

    @overload
    def __getitem__(self, keywords: tuple[()]) -> StandaloneData: ...

    @overload
    def __getitem__(
        self, keywords: tuple[str, Unpack[tuple[str, ...]]]
    ) -> Data | EllipsisType | None: ...

    @override
    def __getitem__(
        self, keywords: tuple[str, ...]
    ) -> Data | StandaloneData | EllipsisType | None:  # ty: ignore[invalid-method-override]
        entry = self._parsed[keywords]
        return entry.data

    @override
    def __setitem__(  # ty: ignore[invalid-method-override]
        self, key: Never, value: Never
    ) -> None:  # pragma: no cover
        msg = "Use 'put' method instead"
        raise NotImplementedError(msg)

    @overload
    def put(
        self,
        keywords: tuple[()],
        /,
        data: StandaloneData,
        content: bytes,
    ) -> None: ...

    @overload
    def put(
        self,
        keywords: tuple[str, Unpack[tuple[str, ...]]],
        /,
        data: Data | EllipsisType | None,
        content: bytes,
    ) -> None: ...

    def put(
        self,
        keywords: tuple[str, ...],
        /,
        data: Data | StandaloneData | EllipsisType | None,
        content: bytes,
    ) -> None:
        start, end = self.entry_location(keywords)

        self._update_content(start, end, content)
        self._parsed[keywords] = ParsedEntry(data, start, start + len(content))
        self._remove_child_entries(keywords)

    @overload
    def add(
        self,
        keywords: tuple[()],
        data: StandaloneData,
        content: bytes,
        /,
    ) -> None: ...

    @overload
    def add(
        self,
        keywords: tuple[str, Unpack[tuple[str, ...]]],
        data: Data | EllipsisType | None,
        content: bytes,
        /,
    ) -> None: ...

    @override
    def add(  # ty: ignore[invalid-method-override]
        self,
        keywords: tuple[str, ...],
        data: Data | StandaloneData | EllipsisType | None,
        content: bytes,
        /,
    ) -> None:
        assert keywords not in self._parsed or (
            keywords and keywords[-1].startswith("#")
        )
        assert not keywords or not keywords[-1].startswith("#") or data is not ...
        assert keywords or data is not ...

        start, end = self.entry_location(keywords, add=True)

        self._parsed.add(keywords, ParsedEntry(data, start, end))
        self._update_content(start, end, content)

    @overload
    @with_default
    def popone(self, keywords: tuple[()]) -> StandaloneData: ...

    @overload
    @with_default
    def popone(
        self, keywords: tuple[str, Unpack[tuple[str, ...]]]
    ) -> Data | EllipsisType | None: ...

    @override
    @with_default
    def popone(
        self, keywords: tuple[str, ...], /
    ) -> Data | StandaloneData | EllipsisType | None:
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
                assert isinstance(entry, ParsedEntry)
                if entry.start >= end:
                    entry.start += diff
                    entry.end += diff
                elif entry.end > start:
                    entry.end += diff

        self.contents[start:end] = new_content
        self.modified = True

    def _remove_child_entries(self, keywords: tuple[str, ...], /) -> None:
        """Remove all child entries of the given keywords."""
        for k in list(self._parsed):
            if keywords != k and keywords == k[: len(keywords)]:
                del self._parsed[k]

    def entry_location(
        self, keywords: tuple[str, ...], /, *, add: bool = False
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

    def as_dict(self) -> FileDict:
        ret: FileDict = {}
        for keywords, entry in self._parsed.items():
            assert isinstance(entry, ParsedEntry)
            if not keywords:
                assert entry.data is not ...
                assert None not in ret
                ret[None] = entry.data
            elif entry.data is ...:
                parent: FileDict | SubDict = ret
                for k in keywords[:-1]:
                    sub = parent[k]
                    assert isinstance(sub, (dict, MultiDict))
                    parent = sub
                assert keywords[-1] not in parent
                parent[keywords[-1]] = {}
            elif len(keywords) == 1:
                ret = add_to_mapping(ret, keywords[0], entry.data)  # ty: ignore[invalid-assignment]
            else:
                grandparent: FileDict | SubDict = ret
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
