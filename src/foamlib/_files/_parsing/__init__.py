import dataclasses
import sys
from collections.abc import Collection, Iterator, Mapping
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
from pyparsing import TypeVar

from ...typing import Data, FileDict, StandaloneData, SubDict
from .._util import add_to_mapping
from ._exceptions import ParseError
from ._parser import (
    LocatedEntry,
    _expect,
    _find_matching_brace,
    _parse_data_entry,
    parse_data,
    parse_file,
    parse_file_located,
    parse_standalone_data,
    parse_token,
    skip,
)

_T = TypeVar("_T", str, Data, StandaloneData, FileDict)


def parse(contents: bytes | str, /, *, target: type[_T]) -> _T:
    if isinstance(contents, str):
        contents = contents.encode("latin-1")

    if target is str:
        parse = parse_token
    elif target is Data:
        parse = parse_data
    elif target is StandaloneData:
        parse = parse_standalone_data
    elif target is FileDict:
        parse = parse_file
    else:
        msg = f"Unsupported type for parsing: {target}"
        raise TypeError(msg)

    try:
        ret, pos = parse(contents, 0)
        skip(contents, pos, strict=True)
    except ParseError as e:
        msg = str(e)
        raise ValueError(msg) from e

    return ret  # ty: ignore[invalid-return-type]


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
            parse_results = parse_file_located(contents, 0)
        except ParseError as e:
            msg = f"Failed to parse contents: {e}"
            raise ValueError(msg) from e
        self._parsed = self._flatten_results(contents, parse_results)

        self.contents = contents
        self.modified = False

    @staticmethod
    def _flatten_results(
        contents: bytes,
        parse_results: list[LocatedEntry],
        /,
        *,
        _keywords: tuple[str, ...] = (),
    ) -> MultiDict[tuple[str, ...], "ParsedFile._Entry"]:
        ret: MultiDict[tuple[str, ...], ParsedFile._Entry] = MultiDict()
        for parse_result in parse_results:
            keyword, data = parse_result.value
            start = parse_result.locn_start
            end = parse_result.locn_end

            if keyword is None:
                assert not _keywords
                assert () not in ret
                data = cast("StandaloneData", data)
                ret[()] = ParsedFile._Entry(data, start, end)
            else:
                assert isinstance(keyword, str)
                if isinstance(data, dict):
                    # This is a subdictionary - recursively parse its entries
                    if (*_keywords, keyword) in ret:
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    ret[(*_keywords, keyword)] = ParsedFile._Entry(..., start, end)
                    # Parse the subdictionary content to get nested entries with locations
                    dict_start = contents.find(b"{", start)
                    dict_end = contents.rfind(b"}", start, end)
                    if dict_start != -1 and dict_end != -1:
                        subdict_entries = ParsedFile._parse_subdict_located(
                            contents, dict_start + 1, dict_end
                        )
                        ret.extend(
                            ParsedFile._flatten_results(
                                contents,
                                subdict_entries,
                                _keywords=(*_keywords, keyword),
                            )
                        )
                else:
                    if (*_keywords, keyword) in ret and not keyword.startswith("#"):
                        msg = f"Duplicate entry found for keyword: {keyword}"
                        raise ValueError(msg)
                    assert not isinstance(data, Mapping)
                    ret.add((*_keywords, keyword), ParsedFile._Entry(data, start, end))
        return ret

    @staticmethod
    def _parse_subdict_located(
        contents: bytes, start: int, end: int
    ) -> list[LocatedEntry]:
        """Parse entries within a subdictionary and return them with locations."""
        ret: list[LocatedEntry] = []
        pos = start

        while (pos := skip(contents, pos)) < end:
            entry_start = pos
            try:
                keyword, new_pos = parse_token(contents, pos)
                new_pos = skip(contents, new_pos)

                if keyword.startswith("#"):
                    value, new_pos = _parse_data_entry(contents, new_pos)
                    new_pos = skip(contents, new_pos, newline_ok=False)
                    # Expect newline or end of subdictionary for directives
                    if new_pos < end and contents[new_pos : new_pos + 1] == b"\n":
                        new_pos += 1
                # Check if this is a nested subdictionary
                elif contents[new_pos : new_pos + 1] == b"{":
                    # Just find the matching brace without parsing/validating
                    new_pos = _find_matching_brace(contents, new_pos)
                    value = {}  # Marker for subdictionary
                else:
                    try:
                        value, new_pos = parse_data(contents, new_pos)
                    except ParseError:
                        value = None
                    else:
                        new_pos = skip(contents, new_pos)
                    new_pos = _expect(contents, new_pos, b";")

                ret.append(LocatedEntry((keyword, value), entry_start, new_pos))
                pos = new_pos
            except ParseError:
                # End of subdictionary or invalid content
                break

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

    def as_dict(self) -> FileDict:
        ret: FileDict = {}
        for keywords, entry in self._parsed.items():
            assert isinstance(entry, ParsedFile._Entry)
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
