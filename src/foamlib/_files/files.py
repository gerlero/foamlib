import contextlib
import os
import sys
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Literal, TypeVar, cast, overload

if sys.version_info >= (3, 11):
    from typing import Unpack, assert_never
else:
    from typing_extensions import Unpack, assert_never

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import multicollections.abc
import numpy as np
from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping, with_default

from .._files import _common
from ..typing import (
    Data,
    DataLike,
    Field,
    FieldLike,
    FileDict,
    FileDictLike,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
)
from ._io import FoamFileIO
from ._parsing import parse
from ._serialization import dumps, normalized
from ._util import SupportsKeysAndGetItem
from .types import Dimensioned, DimensionSet

_D = TypeVar("_D")


class FoamFile(
    MutableMultiMapping[
        str | None,
        "Data | StandaloneData | FoamFile.SubDict | None",
    ],
    os.PathLike[str],
    FoamFileIO,
):
    """
    An OpenFOAM data file.

    :class:`FoamFile` supports most OpenFOAM data and configuration files (i.e., files with a
    "FoamFile" header), including those with regular expressions and #-based directives.
    Notable exceptions are FoamFiles with #codeStreams, which are currently not supported.
    Non-FoamFile output files are also not supported by this class. Regular expressions and
    #-based directives can be accessed and modified, but they are not evaluated or expanded
    by this library.

    Use :class:`FoamFile` as a mutable mapping (i.e., like a :class:`dict`) to access and modify
    entries. When accessing a sub-dictionary, the returned value will be a
    :class:`FoamFile.SubDict` object, that allows for further access and modification of nested
    dictionaries within the :class:`FoamFile` in a single operation.

    If the :class:`FoamFile` does not store a dictionary, the main stored value can be accessed
    and modified by passing ``None`` as the key (e.g., ``file[None]``).

    You can also use the :class:`FoamFile` as a context manager (i.e., within a ``with`` block)
    to make multiple changes to the file while saving any and all changes only once at
    the end.

    :param path: The path to the file. If the file does not exist, it will be created
        when the first change is made. However, if an attempt is made to access entries
        in a non-existent file, a :class:`FileNotFoundError` will be raised.

    Example usage: ::

        from foamlib import FoamFile

        file = FoamFile("path/to/case/system/controlDict") # Load a controlDict file
        print(file["endTime"]) # Print the end time
        file["writeInterval"] = 100 # Set the write interval to 100
        file["writeFormat"] = "binary" # Set the write format to binary

    or (better): ::

        from foamlib import FoamCase

        case = FoamCase("path/to/case")

        with case.control_dict as file: # Load the controlDict file
            print(file["endTime"]) # Print the end time
            file["writeInterval"] = 100
            file["writeFormat"] = "binary"
    """

    Dimensioned = Dimensioned
    DimensionSet = DimensionSet

    class KeysView(multicollections.abc.KeysView[str | None]):
        def __init__(self, file: "FoamFile", *, include_header: bool = False) -> None:
            self._file = file
            self._include_header = include_header

        @override
        def __iter__(self) -> Iterator[str | None]:
            return self._file._iter(include_header=self._include_header)

        @override
        def __len__(self) -> int:
            parsed = self._file._get_parsed()
            return sum(
                1
                for k in parsed
                if len(k) <= 1 and (k != ("FoamFile",) or self._include_header)
            )

        @override
        def __contains__(self, x: object) -> bool:  # ty: ignore[invalid-method-override]
            if not isinstance(x, str) and x is not None:
                return False
            keywords = (x,) if x is not None else ()
            parsed = self._file._get_parsed()
            if keywords in parsed:
                # Check if this key should be included in the view
                return keywords != ("FoamFile",) or self._include_header
            return False

    class ValuesView(
        multicollections.abc.ValuesView[
            "Data | StandaloneData | FoamFile.SubDict | None"
        ]
    ):
        def __init__(self, file: "FoamFile", *, include_header: bool = False) -> None:
            self._file = file
            self._include_header = include_header

        @override
        def __iter__(
            self,
        ) -> Iterator["Data | StandaloneData | FoamFile.SubDict | None"]:
            for k, v in self._file._get_parsed().items():
                if k != ("FoamFile",) or self._include_header:
                    if v is ...:
                        assert k
                        k = cast("tuple[str, Unpack[tuple[str, ...]]]", k)
                        yield FoamFile.SubDict(self._file, k)
                    else:
                        yield v

        @override
        def __len__(self) -> int:
            parsed = self._file._get_parsed()
            return sum(
                1
                for k in parsed
                if len(k) <= 1 and (k != ("FoamFile",) or self._include_header)
            )

        @override
        def __contains__(self, value: object) -> bool:
            return any(v == value for v in iter(self))

    class ItemsView(
        multicollections.abc.ItemsView[
            str | None, "Data | StandaloneData | FoamFile.SubDict | None"
        ]
    ):
        def __init__(
            self,
            file: "FoamFile",
            *,
            include_header: bool = False,
        ) -> None:
            self._file = file
            self._include_header = include_header

        @override
        def __iter__(
            self,
        ) -> Iterator[
            tuple[str, "Data | FoamFile.SubDict | None"] | tuple[None, StandaloneData]
        ]:
            for k, v in self._file._get_parsed().items():
                if k != ("FoamFile",) or self._include_header:
                    if not k:
                        yield None, v
                    else:
                        k = cast("tuple[str, Unpack[tuple[str, ...]]]", k)
                        yield (
                            k[-1],
                            v if v is not ... else FoamFile.SubDict(self._file, k),
                        )

        @override
        def __len__(self) -> int:
            parsed = self._file._get_parsed()
            return sum(
                1
                for k in parsed
                if len(k) <= 1 and (k != ("FoamFile",) or self._include_header)
            )

        @override
        def __contains__(self, x: object) -> bool:  # ty: ignore[invalid-method-override]
            return any(i == x for i in iter(self))

    class SubDict(
        MutableMultiMapping[str, "Data | FoamFile.SubDict | None"],
    ):
        """
        An OpenFOAM sub-dictionary within a file.

        :class:`FoamFile.SubDict` is a mutable mapping that allows for accessing and modifying
        nested dictionaries within a :class:`FoamFile` in a single operation. It behaves like a
        :class:`dict` and can be used to access and modify entries in the sub-dictionary.

        To obtain a :class:`FoamFile.SubDict` object, access a sub-dictionary in a :class:`FoamFile`
        object (e.g., ``file["subDict"]``).

        Example usage: ::

            from foamlib import FoamFile

            file = FoamFile("path/to/case/system/fvSchemes") # Load an fvSchemes file
            print(file["ddtSchemes"]["default"]) # Print the default ddt scheme
            file["ddtSchemes"]["default"] = "Euler" # Set the default ddt scheme

        or (better): ::

            from foamlib import FoamCase

            case = FoamCase("path/to/case")

            with case.fv_schemes as file: # Load the fvSchemes file
                print(file["ddtSchemes"]["default"])
                file["ddtSchemes"]["default"] = "Euler"
        """

        class KeysView(multicollections.abc.KeysView[str]):
            def __init__(self, subdict: "FoamFile.SubDict") -> None:
                self._subdict = subdict

            @override
            def __iter__(self) -> Iterator[str]:
                return self._subdict._file._iter(keywords=self._subdict._keywords)

            @override
            def __len__(self) -> int:
                parsed = self._subdict._file._get_parsed()
                return sum(1 for k in parsed if k[:-1] == self._subdict._keywords)

            @override
            def __contains__(self, x: object) -> bool:  # ty: ignore[invalid-method-override]
                if not isinstance(x, str):
                    return False
                return (
                    *self._subdict._keywords,
                    x,
                ) in self._subdict._file._get_parsed()

        class ValuesView(
            multicollections.abc.ValuesView["Data | FoamFile.SubDict | None"]
        ):
            def __init__(self, subdict: "FoamFile.SubDict") -> None:
                self._subdict = subdict

            @override
            def __iter__(self) -> Iterator["Data | FoamFile.SubDict | None"]:
                for k, v in self._subdict._file._get_parsed().items():
                    if k[:-1] == self._subdict._keywords:
                        k = cast("tuple[str, Unpack[tuple[str, ...]]]", k)
                        yield (
                            v
                            if v is not ...
                            else FoamFile.SubDict(self._subdict._file, k)
                        )

            @override
            def __len__(self) -> int:
                parsed = self._subdict._file._get_parsed()
                return sum(1 for k in parsed if k[:-1] == self._subdict._keywords)

            @override
            def __contains__(self, value: object) -> bool:
                return any(v == value for v in iter(self))

        class ItemsView(
            multicollections.abc.ItemsView[str, "Data | FoamFile.SubDict | None"]
        ):
            def __init__(self, subdict: "FoamFile.SubDict") -> None:
                self._subdict = subdict

            @override
            def __iter__(
                self,
            ) -> Iterator[tuple[str, "Data | FoamFile.SubDict | None"]]:
                for k, v in self._subdict._file._get_parsed().items():
                    if k[:-1] == self._subdict._keywords:
                        k = cast("tuple[str, Unpack[tuple[str, ...]]]", k)
                        yield (
                            k[-1],
                            v
                            if v is not ...
                            else FoamFile.SubDict(self._subdict._file, k),
                        )

            @override
            def __len__(self) -> int:
                parsed = self._subdict._file._get_parsed()
                return sum(1 for k in parsed if k[:-1] == self._subdict._keywords)

            @override
            def __contains__(self, x: object) -> bool:  # ty: ignore[invalid-method-override]
                return any(i == x for i in iter(self))

        def __init__(
            self, _file: "FoamFile", _keywords: tuple[str, Unpack[tuple[str, ...]]]
        ) -> None:
            self._file = _file
            self._keywords = _keywords

        @override
        @with_default
        def getall(
            self, keyword: str, /
        ) -> Collection["Data | FoamFile.SubDict | None"]:
            return self._file.getall((*self._keywords, keyword))  # ty: ignore[invalid-return-type]

        @override
        def __getitem__(self, keyword: str) -> "Data | FoamFile.SubDict | None":
            return self._file[(*self._keywords, keyword)]  # ty: ignore[invalid-return-type]

        @overload
        def __setitem__(
            self, keyword: str, data: DataLike | SubDictLike | None
        ) -> None: ...

        @overload
        def __setitem__(self, keyword: slice, data: SubDictLike) -> None: ...

        @override
        def __setitem__(
            self, keyword: str | slice, data: DataLike | SubDictLike | None
        ) -> None:
            if keyword == slice(None):
                if not isinstance(data, Mapping):
                    msg = "Can only set entire SubDict with a mapping"
                    raise TypeError(msg)
                with self._file:
                    self.clear()
                    self.extend(data)  # type: ignore[invalid-argument-type]
                return

            if isinstance(keyword, slice):
                msg = "Only empty slices (:) are supported"
                raise ValueError(msg)  # noqa: TRY004

            self._file[(*self._keywords, keyword)] = data

        @override
        def add(self, keyword: str, data: DataLike | SubDictLike | None) -> None:
            self._file.add((*self._keywords, keyword), data)

        @override
        @with_default
        def popone(self, keyword: str, /) -> "Data | SubDict | None":
            with self._file:
                ret = self._file[(*self._keywords, keyword)]
                if isinstance(ret, FoamFile.SubDict):
                    ret = ret.as_dict()
                else:
                    ret = deepcopy(ret)
                self._file.popone((*self._keywords, keyword))

            return ret  # ty: ignore[invalid-return-type]

        @override
        def __delitem__(self, keyword: str | slice) -> None:
            if keyword == slice(None):
                self.clear()
                return

            if isinstance(keyword, slice):
                msg = "Only empty slices (:) are supported"
                raise ValueError(msg)  # noqa: TRY004

            del self._file[(*self._keywords, keyword)]

        @override
        def __iter__(self) -> Iterator[str]:
            for k in self._file._iter(self._keywords):
                assert k is not None
                yield k

        @override
        def __contains__(self, keyword: object) -> bool:
            if not isinstance(keyword, str):
                return False
            return (*self._keywords, keyword) in self._file._get_parsed()

        @override
        def __len__(self) -> int:
            parsed = self._file._get_parsed()
            return sum(1 for k in parsed if k[:-1] == self._keywords)

        @override
        def keys(self) -> "FoamFile.SubDict.KeysView":
            return FoamFile.SubDict.KeysView(self)

        @override
        def values(self) -> "FoamFile.SubDict.ValuesView":
            return FoamFile.SubDict.ValuesView(self)

        @override
        def items(self) -> "FoamFile.SubDict.ItemsView":
            return FoamFile.SubDict.ItemsView(self)

        @override
        def update(  # ty: ignore[invalid-method-override]
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike | None]
            | Iterable[tuple[str, DataLike | SubDictLike | None]] = (),
            /,
            **kwargs: DataLike | SubDictLike | None,
        ) -> None:
            with self._file:
                super().update(other, **kwargs)  # ty: ignore[invalid-argument-type]

        @override
        def extend(  # ty: ignore[invalid-method-override]
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike | None]
            | Iterable[tuple[str, DataLike | SubDictLike | None]] = (),
            /,
            **kwargs: DataLike | SubDictLike | None,
        ) -> None:
            with self._file:
                super().extend(other, **kwargs)  # ty: ignore[invalid-argument-type]

        @override
        def merge(  # ty: ignore[invalid-method-override]
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike | None]
            | Iterable[tuple[str, DataLike | SubDictLike | None]] = (),
            /,
            **kwargs: DataLike | SubDictLike | None,
        ) -> None:
            with self._file:
                super().merge(other, **kwargs)  # ty: ignore[invalid-argument-type]

        @override
        def clear(self) -> None:
            with self._file:
                super().clear()

        @override
        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self._file}', {self._keywords})"

        def as_dict(self) -> SubDict:
            """Return a nested dict representation of the sub-dictionary."""
            file = self._file.as_dict(include_header=True)

            ret = file[self._keywords[0]]
            assert isinstance(ret, Mapping)
            for k in self._keywords[1:]:
                ret = ret[k]
                assert isinstance(ret, Mapping)

            return ret  # ty: ignore[invalid-return-type]

    @property
    def version(self) -> float:
        """Alias of ``self["FoamFile"]["version"]``."""
        ret = self["FoamFile", "version"]
        if not isinstance(ret, (int, float)):
            msg = "version is not a number"
            raise TypeError(msg)
        return ret

    @version.setter
    def version(self, value: float) -> None:
        self["FoamFile", "version"] = value

    @property
    def format(self) -> Literal["ascii", "binary"]:
        """Alias of ``self["FoamFile"]["format"]``."""
        ret = self["FoamFile", "format"]
        if not isinstance(ret, str):
            msg = "format is not a string"
            raise TypeError(msg)
        if ret not in ("ascii", "binary"):
            msg = "format is not 'ascii' or 'binary'"
            raise ValueError(msg)
        return cast("Literal['ascii', 'binary']", ret)

    @format.setter
    def format(self, value: Literal["ascii", "binary"]) -> None:
        self["FoamFile", "format"] = value

    @property
    def class_(self) -> str:
        """Alias of ``self["FoamFile"]["class"]``."""
        ret = self["FoamFile", "class"]
        if not isinstance(ret, str):
            msg = "class is not a string"
            raise TypeError(msg)
        return ret

    @class_.setter
    def class_(self, value: str) -> None:
        self["FoamFile", "class"] = value

    @property
    def location(self) -> str:
        """Alias of ``self["FoamFile"]["location"]``."""
        ret = self["FoamFile", "location"]
        if not isinstance(ret, str):
            msg = "location is not a string"
            raise TypeError(msg)
        return ret

    @location.setter
    def location(self, value: str) -> None:
        self["FoamFile", "location"] = value

    @property
    def object_(self) -> str:
        """Alias of ``self["FoamFile"]["object"]``."""
        ret = self["FoamFile", "object"]
        if not isinstance(ret, str):
            msg = "object is not a string"
            raise TypeError(msg)
        return ret

    @object_.setter
    def object_(self, value: str) -> None:
        self["FoamFile", "object"] = value

    def _write_header_if_needed(self, keywords: tuple[str, ...]) -> None:
        """Write FoamFile header if needed."""
        try:
            write_header = (
                not self and "FoamFile" not in self and keywords != ("FoamFile",)
            )
        except FileNotFoundError:
            write_header = keywords != ("FoamFile",)

        if write_header:
            self["FoamFile"] = {}
            self.version = 2.0
            self.format = "ascii"
            self.class_ = "dictionary"
            self.location = f'"{self.path.parent.name}"'
            self.object_ = self.path.name.removesuffix(".gz")

    @overload
    def _update_class_for_field_if_needed(
        self,
        keywords: tuple[str, Unpack[tuple[str, ...]]],
        data: Data | SubDict | None,
        /,
    ) -> None: ...

    @overload
    def _update_class_for_field_if_needed(
        self, keywords: tuple[()], data: StandaloneData, /
    ) -> None: ...

    def _update_class_for_field_if_needed(
        self, keywords: tuple[str, ...], data: Data | StandaloneData | SubDict, /
    ) -> None:
        """Update class field to appropriate field type if this is a field entry."""
        try:
            class_ = self.class_
        except (KeyError, FileNotFoundError):
            class_ = None

        match class_, data, keywords:
            case (
                "dictionary",
                float() | np.ndarray(),
                _common.FIELD_KEYWORDS,
            ):
                self.class_ = FoamFile._vol_field_class(data)  # ty: ignore[invalid-argument-type]

    def _calculate_spacing(
        self,
        keywords: tuple[str, ...],
        start: int,
        end: int,
        /,
        *,
        add: bool,
    ) -> tuple[bytes, bytes]:
        """Calculate before/after spacing for entry operations."""
        parsed = self._get_parsed(missing_ok=True)

        # For setitem operations, check if this is an update to an existing entry
        # and preserve existing spacing for sub-dictionary entries
        if not add:
            is_update = keywords in parsed
            if is_update and len(keywords) > 1:
                # For existing sub-dictionary entries, preserve existing formatting
                # The content before 'start' already includes both spacing and indentation,
                # so we don't add any leading newlines to prevent blank line accumulation
                before = b""
            elif start and not parsed.contents[:start].endswith(b"\n\n"):
                if parsed.contents[:start].endswith(b"\n"):
                    before = b"\n" if len(keywords) <= 1 else b""
                else:
                    before = b"\n\n" if len(keywords) <= 1 else b"\n"
            else:
                before = b""
        # Add operations use simpler spacing logic
        elif start and not parsed.contents[:start].endswith(b"\n\n"):
            if parsed.contents[:start].endswith(b"\n"):
                before = b"\n" if len(keywords) <= 1 else b""
            else:
                before = b"\n\n" if len(keywords) <= 1 else b"\n"
        else:
            before = b""

        # Calculate after spacing
        # For updates to existing subdictionary entries, preserve trailing whitespace if present
        if not add and keywords in parsed and len(keywords) > 1:
            # Check if the existing entry ends with a newline
            entry_start, entry_end = parsed.entry_location(keywords)
            existing_content = parsed.contents[entry_start:entry_end]
            # If the existing entry ends with a newline, preserve it
            if existing_content and existing_content[-1:] == b"\n":
                after = b"\n"
            else:
                after = b""
        elif not parsed.contents[end:].strip() or parsed.contents[end:].startswith(
            b"}"
        ):
            after = b"\n" + b"    " * (len(keywords) - 2)
        else:
            after = b""

        return before, after

    @overload
    def _perform_entry_operation(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        data: DataLike | SubDictLike | None,
        /,
        *,
        add: bool,
    ) -> None: ...

    @overload
    def _perform_entry_operation(
        self,
        keywords: None | tuple[()],
        data: StandaloneDataLike,
        /,
        *,
        add: bool,
    ) -> None: ...

    def _perform_entry_operation(
        self,
        keywords: tuple[str, ...],
        data: DataLike | StandaloneDataLike | SubDictLike | None,
        /,
        *,
        add: bool,
    ) -> None:
        """Shared method for performing entry operations (setitem and add)."""
        if keywords:
            keyword = keywords[-1]

            if not isinstance(keyword, str):
                msg = (
                    f"Invalid keyword type: {keywords[-1]} (type {type(keywords[-1])})"
                )
                raise TypeError(msg)

            if keyword != parse(keyword, target=str):
                msg = f"Invalid keyword string: {keywords[-1]!r}"
                raise ValueError(msg)

        data = normalized(data, keywords=keywords)  # ty: ignore[no-matching-overload]

        indentation = b"    " * (len(keywords) - 1)

        with self:
            self._write_header_if_needed(keywords)
            self._update_class_for_field_if_needed(keywords, data)  # ty: ignore[invalid-argument-type]

            parsed = self._get_parsed(missing_ok=True)
            start, end = parsed.entry_location(keywords, add=add)
            before, after = self._calculate_spacing(keywords, start, end, add=add)

            try:
                format_ = self.format
            except (KeyError, FileNotFoundError):
                format_ = None

            if isinstance(data, Mapping):
                if not keywords:
                    msg = "Cannot set a mapping at the root level of a FoamFile\nUse update(), extend(), or merge() instead."
                    raise ValueError(msg)
                keywords = cast("tuple[str, Unpack[tuple[str, ...]]]", keywords)

                if keyword.startswith("#"):
                    msg = f"Cannot set a directive as the keyword for a dictionary: {keyword}"
                    raise ValueError(msg)

                if add and keywords in parsed:
                    raise KeyError(keywords)

                empty_dict_content = (
                    before
                    + indentation
                    + dumps(keyword)
                    + b"\n"
                    + indentation
                    + b"{\n"
                    + indentation
                    + b"}"
                    + after
                )
                parsed.put(keywords, ..., empty_dict_content)

                for k, v in data.items():
                    self[(*keywords, k)] = v  # ty: ignore[invalid-assignment]

            elif keywords:
                keywords = cast("tuple[str, Unpack[tuple[str, ...]]]", keywords)
                val = dumps(data, keywords=keywords, format_=format_)

                # When updating existing subdictionary entries, check if the existing entry
                # includes indentation in its boundaries. If so, we need to preserve it.
                # Note: is_update is also calculated in _calculate_spacing for determining 'before'
                is_update = not add and keywords in parsed
                if is_update and len(keywords) > 1:
                    # Check if the existing entry starts with indentation
                    entry_start, entry_end = parsed.entry_location(keywords)
                    existing_content = parsed.contents[entry_start:entry_end]
                    # If the existing entry starts with whitespace, preserve it
                    if existing_content and existing_content[0:1] in (b" ", b"\t"):
                        # Extract leading whitespace from existing entry
                        leading_ws_len = 0
                        for byte in existing_content:
                            if byte in (ord(b" "), ord(b"\t")):  # space or tab
                                leading_ws_len += 1
                            else:
                                break
                        content_indentation = bytes(existing_content[:leading_ws_len])
                    else:
                        content_indentation = b""
                else:
                    content_indentation = indentation

                content = (
                    before
                    + content_indentation
                    + dumps(keyword)
                    + ((b" " + val) if val else b"")
                    + (b";" if not keywords[-1].startswith("#") else b"")
                    + after
                )

                if add:
                    if keywords in parsed and not keywords[-1].startswith("#"):
                        raise KeyError(keywords)

                    parsed.add(keywords, data, content)
                else:
                    parsed.put(keywords, data, content)

            else:
                if add and () in parsed:
                    raise KeyError(None)

                content = before + dumps(data, keywords=(), format_=format_) + after

                parsed.put((), data, content)

    @overload
    @with_default
    def getall(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        /,
    ) -> Collection["Data | FoamFile.SubDict | None"]: ...

    @overload
    @with_default
    def getall(self, keywords: None | tuple[()], /) -> Collection[StandaloneData]: ...

    @override
    @with_default
    def getall(
        self,
        keywords: str | tuple[str, ...] | None,
        /,
    ) -> Collection["Data | StandaloneData | FoamFile.SubDict | None"]:
        keywords = FoamFile._normalized_keywords(keywords)
        keywords = cast("tuple[str, Unpack[tuple[str, ...]]] | tuple[()]", keywords)

        parsed = self._get_parsed()

        values = parsed.getall(keywords)

        ret = []
        for v in values:
            if v is ...:
                assert keywords
                ret.append(FoamFile.SubDict(self, keywords))
            else:
                ret.append(deepcopy(v))
        return ret

    @overload
    @with_default
    def getone(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        /,
    ) -> "Data | FoamFile.SubDict | None": ...

    @overload
    @with_default
    def getone(self, keywords: None | tuple[()], /) -> StandaloneData: ...

    @override
    @with_default
    def getone(
        self,
        keywords: str | tuple[str, ...] | None,
        /,
    ) -> "Data | StandaloneData | FoamFile.SubDict | None":
        keywords = FoamFile._normalized_keywords(keywords)

        parsed = self._get_parsed()
        keywords = cast("tuple[str, Unpack[tuple[str, ...]]] | tuple[()]", keywords)
        ret = parsed[keywords]
        if ret is ...:
            assert keywords
            return FoamFile.SubDict(self, keywords)
        return deepcopy(ret)

    @overload
    def get(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        default: _D = ...,
        /,
    ) -> "Data | FoamFile.SubDict | None | _D": ...

    @overload
    def get(
        self,
        keywords: None | tuple[()],
        default: _D = ...,
        /,
    ) -> StandaloneData | _D: ...

    @override
    def get(  # ty: ignore[invalid-method-override]
        self,
        keywords: str | tuple[str, ...] | None,
        default: _D = None,  # ty: ignore[invalid-parameter-default]
        /,
    ) -> "Data | StandaloneData | FoamFile.SubDict | None | _D":
        return self.getone(keywords, default=default)

    @overload
    def __getitem__(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
    ) -> "Data | FoamFile.SubDict | None": ...

    @overload
    def __getitem__(
        self,
        keywords: None | tuple[()],
    ) -> StandaloneData: ...

    @override
    def __getitem__(  # ty: ignore[invalid-method-override]
        self,
        keywords: str | tuple[str, ...] | None,
    ) -> "Data | StandaloneData | FoamFile.SubDict | None":
        return self.getone(keywords)

    @overload
    def __setitem__(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        data: DataLike | SubDictLike | None,
    ) -> None: ...

    @overload
    def __setitem__(
        self, keywords: None | tuple[()], data: StandaloneDataLike
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        keywords: slice,
        data: FileDictLike,
    ) -> None: ...

    @override
    def __setitem__(  # ty: ignore[invalid-method-override]
        self,
        keywords: str | tuple[str, ...] | None | slice,
        data: DataLike | StandaloneDataLike | SubDictLike | None | FileDictLike,
    ) -> None:
        keywords = FoamFile._normalized_keywords(keywords, slice_ok=True)

        if keywords == slice(None):
            if not isinstance(data, Mapping):
                msg = "Can only set the entire FoamFile from a mapping"
                raise TypeError(msg)
            with self:
                with contextlib.suppress(FileNotFoundError):
                    self.clear()
                self.extend(data)  # ty: ignore[invalid-argument-type]
            return

        assert not isinstance(keywords, slice)
        self._perform_entry_operation(keywords, data, add=False)  # ty: ignore[invalid-argument-type,no-matching-overload]

    @override
    def __delitem__(self, keywords: str | tuple[str, ...] | None | slice) -> None:
        keywords = FoamFile._normalized_keywords(keywords, slice_ok=True)

        if keywords == slice(None):
            with self:
                self.clear()
            return

        assert not isinstance(keywords, slice)

        with self:
            del self._get_parsed()[keywords]

    @overload
    def add(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        data: DataLike | SubDictLike | None,
    ) -> None: ...

    @overload
    def add(
        self,
        keywords: None | tuple[()],
        data: StandaloneDataLike,
    ) -> None: ...

    @override
    def add(  # ty: ignore[invalid-method-override]
        self,
        keywords: str | tuple[str, ...] | None,
        data: DataLike | StandaloneDataLike | SubDictLike | None,
    ) -> None:
        keywords = FoamFile._normalized_keywords(keywords)
        keywords = cast("tuple[str, Unpack[tuple[str, ...]]] | tuple[()]", keywords)
        self._perform_entry_operation(keywords, data, add=True)

    @overload
    @with_default
    def popone(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
        /,
    ) -> "Data | SubDict | None": ...

    @overload
    @with_default
    def popone(self, keywords: None | tuple[()], /) -> StandaloneData: ...

    @override
    @with_default
    def popone(
        self,
        keywords: str | tuple[str, ...] | None,
        /,
    ) -> "Data | StandaloneData | SubDict | None":
        keywords = FoamFile._normalized_keywords(keywords)

        with self:
            ret = self._get_parsed()[keywords]
            if ret is ...:
                assert keywords
                ret = FoamFile.SubDict(self, keywords).as_dict()
            else:
                ret = deepcopy(ret)
            self._get_parsed().popone(keywords)

        return ret  # ty: ignore[invalid-return-type]

    @overload
    def _iter(
        self,
        keywords: tuple[str, Unpack[tuple[str, ...]]],
        *,
        include_header: bool = ...,
    ) -> Iterator[str]: ...

    @overload
    def _iter(
        self, keywords: tuple[()] = ..., *, include_header: bool = ...
    ) -> Iterator[str | None]: ...

    def _iter(
        self, keywords: tuple[str, ...] = (), *, include_header: bool = False
    ) -> Iterator[str | None]:
        yield from (
            k[-1] if k else None
            for k in self._get_parsed()
            if k[:-1] == keywords and (k != ("FoamFile",) or include_header)
        )

    @override
    def __iter__(self) -> Iterator[str | None]:
        """Iterate over the top-level keys in the FoamFile (excluding the FoamFile header if present)."""
        return self._iter()

    @override
    def __contains__(self, keywords: object) -> bool:
        """Check if the FoamFile contains the given keyword or tuple of keywords."""
        try:
            keywords = FoamFile._normalized_keywords(keywords)  # ty: ignore[no-matching-overload]
        except (ValueError, TypeError):
            return False

        return keywords in self._get_parsed()

    @override
    def __len__(self) -> int:
        """Return the number of top-level keywords in the FoamFile (excluding the FoamFile header if present)."""
        parsed = self._get_parsed()
        return sum(1 for k in parsed if len(k) <= 1 and k != ("FoamFile",))

    @override
    def keys(
        self,
        *,
        include_header: bool = False,
    ) -> "FoamFile.KeysView":
        """
        Return a collection of the keywords in the FoamFile.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        return FoamFile.KeysView(self, include_header=include_header)

    @override
    def values(
        self,
        *,
        include_header: bool = False,
    ) -> "FoamFile.ValuesView":
        """
        Return a collection of the values in the FoamFile.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        return FoamFile.ValuesView(self, include_header=include_header)

    @override
    def items(
        self,
        *,
        include_header: bool = False,
    ) -> "FoamFile.ItemsView":
        """
        Return a collection of the items (keyword-value pairs) in the FoamFile.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        return FoamFile.ItemsView(self, include_header=include_header)

    @override
    def update(  # ty: ignore[invalid-method-override]
        self,
        other: SupportsKeysAndGetItem[
            str | None,
            DataLike | StandaloneDataLike | SubDictLike | None,
        ]
        | Iterable[
            tuple[
                str | None,
                DataLike | StandaloneDataLike | SubDictLike | None,
            ]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike | None,
    ) -> None:
        with self:
            super().update(other, **kwargs)  # ty: ignore[invalid-argument-type]

    @override
    def extend(  # ty: ignore[invalid-method-override]
        self,
        other: SupportsKeysAndGetItem[
            str | None,
            DataLike | StandaloneDataLike | SubDictLike | None,
        ]
        | Iterable[
            tuple[
                str | None,
                DataLike | StandaloneDataLike | SubDictLike | None,
            ]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike | None,
    ) -> None:
        with self:
            super().extend(other, **kwargs)  # ty: ignore[invalid-argument-type]

    @override
    def merge(  # ty: ignore[invalid-method-override]
        self,
        other: SupportsKeysAndGetItem[
            str | None, DataLike | StandaloneDataLike | SubDictLike | None
        ]
        | Iterable[
            tuple[str | None, DataLike | StandaloneDataLike | SubDictLike | None]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike | None,
    ) -> None:
        with self:
            super().merge(other, **kwargs)  # ty: ignore[invalid-argument-type]

    @override
    def clear(self, include_header: bool = False) -> None:
        """
        Remove all entries from the FoamFile.

        :param include_header: Whether to also remove the "FoamFile" header.
        """
        with self:
            if include_header:
                self._get_parsed().clear()
            else:
                super().clear()

    @override
    def __fspath__(self) -> str:
        return str(self.path)

    def as_dict(self, *, include_header: bool = False) -> FileDict:
        """
        Return a nested dict representation of the file.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        d = self._get_parsed().as_dict()
        if not include_header:
            d.pop("FoamFile", None)
        return deepcopy(d)

    @staticmethod
    def loads(
        s: bytes | bytearray | str,
        *,
        include_header: bool = False,
    ) -> FileDict | StandaloneData:
        """
        Standalone deserializing function.

        Deserialize the OpenFOAM FoamFile format to Python objects.

        :param s: The string to deserialize. This can be a dictionary, list, or any
            other object that can be serialized to the OpenFOAM format.
        :param include_header: Whether to include the "FoamFile" header in the output.
            If `True`, the header will be included if it is present in the input object.
        """
        file = parse(s, target=FileDict)  # ty: ignore[invalid-argument-type]

        ret = (
            cast("StandaloneData", file[None])
            if len(file) == 1 and None in file
            else file
        )

        if not include_header and isinstance(ret, Mapping) and "FoamFile" in ret:
            del ret["FoamFile"]
            if len(ret) == 1 and None in ret:
                val = ret[None]
                assert not isinstance(val, Mapping)
                return val

        return ret

    @staticmethod
    def dumps(
        file: FileDictLike | StandaloneDataLike, *, ensure_header: bool = True
    ) -> bytes:
        """
        Standalone serializing function.

        Serialize Python objects to the OpenFOAM FoamFile format.

        :param file: The Python object to serialize. This can be a dictionary, list,
            or any other object that can be serialized to the OpenFOAM format.
        :param ensure_header: Whether to include the "FoamFile" header in the output.
            If ``True``, a header will be included if it is not already present in the
            input object.
        """
        file = normalized(file)

        if not isinstance(file, Mapping):
            file = {None: file}  # ty: ignore[invalid-assignment]

        if "FoamFile" not in file and ensure_header:  # ty: ignore[invalid-argument-type,unsupported-operator]
            class_ = "dictionary"
            try:
                internal_field = file["internalField"]  # ty: ignore[invalid-argument-type,non-subscriptable]
            except KeyError:
                pass
            else:
                if isinstance(internal_field, (float, np.ndarray)):
                    class_ = FoamFile._vol_field_class(internal_field)  # ty: ignore[invalid-argument-type]

            new = MultiDict(
                FoamFile={"version": 2.0, "format": "ascii", "class": class_}
            )
            new.extend(file)  # ty: ignore[invalid-argument-type]
            file = new

        return dumps(file, keywords=())  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

    @overload
    @staticmethod
    def _normalized_keywords(
        keywords: str, /, *, slice_ok: bool = ...
    ) -> tuple[str]: ...

    @overload
    @staticmethod
    def _normalized_keywords(
        keywords: tuple[str, Unpack[tuple[str, ...]]], /, *, slice_ok: bool = ...
    ) -> tuple[str, Unpack[tuple[str, ...]]]: ...

    @overload
    @staticmethod
    def _normalized_keywords(
        keywords: None | tuple[()], /, *, slice_ok: bool = ...
    ) -> tuple[()]: ...

    @overload
    @staticmethod
    def _normalized_keywords(
        keywords: slice, /, *, slice_ok: Literal[True] = ...
    ) -> slice: ...

    @staticmethod
    def _normalized_keywords(
        keywords: tuple[str, ...] | None | slice, /, *, slice_ok: bool = False
    ) -> tuple[str, ...] | slice:
        match keywords:
            case None:
                return ()
            case str():
                return (keywords,)
            case tuple((*_,)) if all(isinstance(k, str) for k in keywords):  # ty: ignore[not-iterable]
                return tuple(keywords)  # ty: ignore[invalid-argument-type,invalid-return-type]
            case slice(start=None, stop=None, step=None) if slice_ok:
                return slice(None)
            case slice() if slice_ok:
                msg = "Only empty slices (:) are supported"
                raise ValueError(msg)
            case _:
                msg = f"Invalid keyword type: {keywords!r}"
                raise TypeError(msg)

    @staticmethod
    def _vol_field_class(field: Field, /) -> str:
        match field:
            case np.ndarray(shape=(3,) | (_, 3)):
                return "volVectorField"
            case np.ndarray(shape=(6,) | (_, 6)):
                return "volSymmTensorField"
            case np.ndarray(shape=(9,) | (_, 9)):
                return "volTensorField"
            case float() | np.ndarray(shape=(_,)):
                return "volScalarField"
            case _:
                assert_never(field)  # ty: ignore[type-assertion-failure]


class FoamFieldFile(FoamFile):
    """
    Subclass of :class:`FoamFile` for representing OpenFOAM field files specifically.

    The difference between :class:`FoamFieldFile` and :class:`FoamFile` is that :class:`FoamFieldFile` has
    the additional properties :attr:`dimensions`, :attr:`internal_field`, and :attr:`boundary_field` that
    are commonly found in OpenFOAM field files. Note that these are only a shorthand for
    accessing the corresponding entries in the file.

    See :class:`FoamFile` for more information on how to read and edit OpenFOAM files.

    :param path: The path to the file. If the file does not exist, it will be created
        when the first change is made. However, if an attempt is made to access entries
        in a non-existent file, a :class:`FileNotFoundError` will be raised.

    Example usage: ::

        from foamlib import FoamFieldFile

        field = FoamFieldFile("path/to/case/0/U") # Load a field
        print(field.dimensions) # Print the dimensions
        print(field.boundary_field) # Print the boundary field
        field.internal_field = [0, 0, 0] # Set the internal field

    or (better): ::

        from foamlib import FoamCase

        case = FoamCase("path/to/case")

        with case[0]["U"] as field: # Load a field
            print(field.dimensions)
            print(field.boundary_field)
            field.internal_field = [0, 0, 0]
    """

    class BoundariesSubDict(FoamFile.SubDict):
        @override
        @with_default
        def getall(
            self, keyword: str, /
        ) -> Collection["FoamFieldFile.BoundarySubDict | Data | None"]:
            ret = super().getall(keyword)
            for r in ret:
                if isinstance(r, FoamFile.SubDict):
                    assert isinstance(r, FoamFieldFile.BoundarySubDict)
            return ret  # ty: ignore[invalid-return-type]

    class BoundarySubDict(FoamFile.SubDict):
        """An OpenFOAM dictionary representing a boundary condition as a mutable mapping."""

        @property
        def type(self) -> str:
            """Alias of ``self["type"]``."""
            ret = self["type"]
            if not isinstance(ret, str):
                msg = "type is not a string"
                raise TypeError(msg)
            return ret

        @type.setter
        def type(self, data: str) -> None:
            self["type"] = data

        @property
        def value(
            self,
        ) -> Field:
            """Alias of ``self["value"]``."""
            return cast(
                "Field",
                self["value"],
            )

        @value.setter
        def value(
            self,
            value: FieldLike,
        ) -> None:
            self["value"] = value

        @value.deleter
        def value(self) -> None:
            del self["value"]

    @overload
    @with_default
    def getall(
        self, keywords: str | tuple[str, Unpack[tuple[str, ...]]], /
    ) -> Collection["Data | FoamFieldFile.SubDict | None"]: ...

    @overload
    @with_default
    def getall(self, keywords: None | tuple[()], /) -> Collection[StandaloneData]: ...

    @override
    @with_default
    def getall(
        self, keywords: str | tuple[str, ...] | None, /
    ) -> Collection["Data | StandaloneData | FoamFieldFile.SubDict | None"]:
        keywords = FoamFieldFile._normalized_keywords(keywords)

        ret = list(super().getall(keywords))

        if keywords[0] == "boundaryField":
            for i, r in enumerate(ret):
                if isinstance(r, FoamFile.SubDict):
                    if len(keywords) == 1:
                        keywords = cast("tuple[str]", keywords)
                        ret[i] = FoamFieldFile.BoundariesSubDict(self, keywords)
                    elif len(keywords) == 2:
                        keywords = cast("tuple[str, str]", keywords)
                        ret[i] = FoamFieldFile.BoundarySubDict(self, keywords)

        return ret

    @overload
    def __getitem__(
        self,
        keywords: str | tuple[str, Unpack[tuple[str, ...]]],
    ) -> "Data | FoamFieldFile.SubDict | None": ...

    @overload
    def __getitem__(
        self,
        keywords: None | tuple[()],
    ) -> StandaloneData: ...

    @override
    def __getitem__(  # ty: ignore[invalid-method-override]
        self,
        keywords: str | tuple[str, ...] | None,
    ) -> "Data | StandaloneData | FoamFieldFile.SubDict | None":
        keywords = FoamFieldFile._normalized_keywords(keywords)

        ret = super().__getitem__(keywords)  # ty: ignore[no-matching-overload]

        if (
            keywords
            and keywords[0] == "boundaryField"
            and isinstance(ret, FoamFile.SubDict)
        ):
            if len(keywords) == 1:
                keywords = cast("tuple[str]", keywords)
                ret = FoamFieldFile.BoundariesSubDict(self, keywords)
            elif len(keywords) == 2:
                keywords = cast("tuple[str, str]", keywords)
                ret = FoamFieldFile.BoundarySubDict(self, keywords)

        return ret

    @property
    def dimensions(self) -> DimensionSet:
        """Alias of ``self["dimensions"]``."""
        ret = self["dimensions"]
        if not isinstance(ret, DimensionSet):
            msg = "dimensions is not a DimensionSet"
            raise TypeError(msg)
        return ret

    @dimensions.setter
    def dimensions(self, value: DimensionSet | Sequence[float]) -> None:
        self["dimensions"] = value

    @property
    def internal_field(
        self,
    ) -> Field:
        """Alias of ``self["internalField"]``."""
        return cast("Field", self["internalField"])

    @internal_field.setter
    def internal_field(
        self,
        value: FieldLike,
    ) -> None:
        self["internalField"] = value

    @property
    def boundary_field(self) -> "FoamFieldFile.BoundariesSubDict":
        """Alias of ``self["boundaryField"]``."""
        ret = self["boundaryField"]
        if not isinstance(ret, FoamFieldFile.BoundariesSubDict):
            assert not isinstance(ret, FoamFile.SubDict)
            msg = "boundaryField is not a dictionary"
            raise TypeError(msg)
        return ret

    @boundary_field.setter
    def boundary_field(self, value: Mapping[str, SubDictLike]) -> None:
        self["boundaryField"] = value
