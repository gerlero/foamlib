import dataclasses
import sys
from collections.abc import Collection, Iterator, Sequence
from typing import Any, TypeVar, cast, overload

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

from .._typing import Data, File, StandaloneData, SubDict
from .._util import add_to_mapping
from ._regex_parser import (
    ParseError,
    parse_data,
    parse_file,
    parse_file_with_locations,
    parse_standalone_data,
    parse_token,
)

_T = TypeVar("_T", str, Data, StandaloneData, File)


def parse(s: bytes | str, /, *, target: type[_T]) -> _T:
    if isinstance(s, str):
        s = s.encode("latin-1")

    if target is str:
        parse_func = parse_token
    elif target is Data:
        parse_func = parse_data
    elif target is StandaloneData:
        parse_func = parse_standalone_data
    elif target is File:
        parse_func = parse_file
    else:
        msg = f"Unsupported type for parsing: {target}"
        raise TypeError(msg)

    try:
        ret = parse_func(s)
    except ParseError as e:
        msg = f"Failed to parse {target}: {e}"
        raise ValueError(msg) from e

    return ret


class ParsedFile(
    MutableMultiMapping[tuple[str, ...], Data | StandaloneData | EllipsisType | None]
):
    @dataclasses.dataclass
    class _Entry:
        data: Data | StandaloneData | EllipsisType | None
        start: int
        end: int

    def __init__(self, contents: bytes | str, /) -> None:
        if isinstance(contents, str):
            contents = contents.encode("latin-1")

        try:
            parse_results = parse_file_with_locations(contents)
        except ParseError as e:
            msg = f"Failed to parse contents: {e}"
            raise ValueError(msg) from e
        self._parsed = self._flatten_results(parse_results)

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_results(
        parse_results: list[tuple[tuple[str | None, Any], int, int]],
        /,
        *,
        _keywords: tuple[str, ...] = (),
    ) -> MultiDict[tuple[str, ...], "ParsedFile._Entry"]:
        ret: MultiDict[tuple[str, ...], ParsedFile._Entry] = MultiDict()
        for (keyword, value), start, end in parse_results:
            if keyword is None:
                assert not _keywords
                assert () not in ret
                ret[()] = ParsedFile._Entry(value, start, end)
            else:
                assert isinstance(keyword, str)
                # Check if value is a dict (nested structure)
                if isinstance(value, (dict, MultiDict)):
                    if (*_keywords, keyword) in ret:
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret[(*_keywords, keyword)] = ParsedFile._Entry(..., start, end)
                    # Flatten nested dict entries
                    for k, v in value.items():
                        nested_key = (*_keywords, keyword, k) if k is not None else (*_keywords, keyword)
                        if isinstance(v, (dict, MultiDict)):
                            ret[nested_key] = ParsedFile._Entry(..., start, end)
                            # Recursively flatten
                            def flatten_dict(d: dict | MultiDict, prefix: tuple[str, ...]) -> None:
                                for k2, v2 in d.items():
                                    key2 = (*prefix, k2) if k2 is not None else prefix
                                    if isinstance(v2, (dict, MultiDict)):
                                        ret[key2] = ParsedFile._Entry(..., start, end)
                                        flatten_dict(v2, key2)
                                    else:
                                        ret.add(key2, ParsedFile._Entry(v2, start, end))
                            flatten_dict(v, nested_key)
                        else:
                            ret.add(nested_key, ParsedFile._Entry(v, start, end))
                else:
                    if (*_keywords, keyword) in ret and not keyword.startswith("#"):
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret.add(
                        (*_keywords, keyword), ParsedFile._Entry(value, start, end)
                    )
        return ret

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
        self._parsed[keywords] = ParsedFile._Entry(data, start, start + len(content))
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

        self._parsed.add(keywords, ParsedFile._Entry(data, start, end))
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
                assert isinstance(entry, ParsedFile._Entry)
                if entry.start >= end:
                    entry.start += diff
                    entry.end += diff
                elif entry.end > start:
                    entry.end += diff

        self.contents = self.contents[:start] + new_content + self.contents[end:]
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

    def as_dict(self) -> File:
        ret: File = {}
        for keywords, entry in self._parsed.items():
            assert isinstance(entry, ParsedFile._Entry)
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
                ret = add_to_mapping(ret, keywords[0], entry.data)  # ty: ignore[invalid-assignment]
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
