from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast, overload

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
else:
    from typing import Collection, Iterable, Iterator, Mapping, Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import multicollections.abc
import numpy as np
from multicollections.abc import (
    MutableMultiMapping,
    with_default,
)

from ._io import FoamFileIO
from ._parsing import Parsed
from ._serialization import dumps, normalize
from .types import Dimensioned, DimensionSet

if TYPE_CHECKING:
    from ._typing import (
        Data,
        DataLike,
        Field,
        FieldLike,
        File,
        FileLike,
        StandaloneData,
        StandaloneDataLike,
        SubDict,
        SubDictLike,
    )
    from ._util import SupportsKeysAndGetItem


def _tensor_kind_for_field(
    field: FieldLike,
) -> str:
    if not (shape := np.shape(field)):
        return "scalar"
    if shape == (3,):
        return "vector"
    if shape == (6,):
        return "symmTensor"
    if shape == (9,):
        return "tensor"
    if len(shape) == 1:
        return "scalar"
    if len(shape) == 2:
        if shape[1] == 3:
            return "vector"
        if shape[1] == 6:
            return "symmTensor"
        if shape[1] == 9:
            return "tensor"

    msg = f"Invalid field shape: {shape}"
    raise ValueError(msg)


class FoamFile(
    MutableMultiMapping[
        "str | tuple[str, ...] | None",
        "Data | StandaloneData | FoamFile.SubDict",
    ],
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

    class KeysView(multicollections.abc.KeysView["str | None"]):
        def __init__(self, file: FoamFile, *, include_header: bool = False) -> None:
            self._file = file
            self._include_header = include_header

        @override
        def __iter__(self) -> Iterator[str | None]:
            return self._file._iter(include_header=self._include_header)

        @override
        def __len__(self) -> int:
            return len(list(iter(self)))

        @override
        def __contains__(self, key: object) -> bool:
            return any(k == key for k in iter(self))

    class ValuesView(
        multicollections.abc.ValuesView["Data | StandaloneData | FoamFile.SubDict"]
    ):
        def __init__(self, file: FoamFile, *, include_header: bool = False) -> None:
            self._file = file
            self._include_header = include_header

        @override
        def __iter__(self) -> Iterator[Data | StandaloneData | FoamFile.SubDict]:
            for k, v in self._file._get_parsed().items():
                if k != ("FoamFile",) or self._include_header:
                    yield v if v is not ... else FoamFile.SubDict(self._file, k)

        @override
        def __len__(self) -> int:
            return len(list(iter(self)))

        @override
        def __contains__(self, value: object) -> bool:
            return value in list(iter(self))

    class ItemsView(
        multicollections.abc.ItemsView[
            "str | None", "Data | StandaloneData | FoamFile.SubDict"
        ]
    ):
        def __init__(
            self,
            file: FoamFile,
            *,
            include_header: bool = False,
            keywords: tuple[str, ...] = (),
        ) -> None:
            assert keywords or include_header
            self._file = file
            self._include_header = include_header
            self._keywords = keywords

        @override
        def __iter__(
            self,
        ) -> Iterator[tuple[str | None, Data | StandaloneData | FoamFile.SubDict]]:
            for k, v in self._file._get_parsed().items():
                if k != ("FoamFile",) or self._include_header:
                    yield (
                        k[-1] if k else None,
                        v if v is not ... else FoamFile.SubDict(self._file, k),
                    )

        @override
        def __len__(self) -> int:
            return len(list(iter(self)))

        @override
        def __contains__(self, item: object) -> bool:
            return item in list(iter(self))

    class SubDict(
        MutableMultiMapping[str, "Data | FoamFile.SubDict"],
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
            def __init__(self, subdict: FoamFile.SubDict) -> None:
                self._subdict = subdict

            @override
            def __iter__(self) -> Iterator[str]:
                return self._subdict._file._iter(keywords=self._subdict._keywords)

            @override
            def __len__(self) -> int:
                return len(list(iter(self)))

            @override
            def __contains__(self, key: object) -> bool:
                return any(k == key for k in iter(self))

        class ValuesView(multicollections.abc.ValuesView["Data | FoamFile.SubDict"]):
            def __init__(self, subdict: FoamFile.SubDict) -> None:
                self._subdict = subdict

            @override
            def __iter__(self) -> Iterator[Data | FoamFile.SubDict]:
                for k, v in self._subdict._file._get_parsed().items():
                    if k[:-1] == self._subdict._keywords:
                        yield (
                            v
                            if v is not ...
                            else FoamFile.SubDict(self._subdict._file, k)
                        )

            @override
            def __len__(self) -> int:
                return len(list(iter(self)))

            @override
            def __contains__(self, value: object) -> bool:
                return any(v == value for v in iter(self))

        class ItemsView(multicollections.abc.ItemsView[str, "Data | FoamFile.SubDict"]):
            def __init__(self, subdict: FoamFile.SubDict) -> None:
                self._subdict = subdict

            @override
            def __iter__(self) -> Iterator[tuple[str, Data | FoamFile.SubDict]]:
                for k, v in self._subdict._file._get_parsed().items():
                    if k[:-1] == self._subdict._keywords:
                        yield (
                            k[-1] if k else None,
                            v
                            if v is not ...
                            else FoamFile.SubDict(self._subdict._file, k),
                        )

            @override
            def __len__(self) -> int:
                return len(list(iter(self)))

            @override
            def __contains__(self, item: object) -> bool:
                return any(i == item for i in iter(self))

        def __init__(self, _file: FoamFile, _keywords: tuple[str, ...]) -> None:
            self._file = _file
            self._keywords = _keywords

        @override
        @with_default
        def getall(self, keyword: str) -> Collection[Data | FoamFile.SubDict]:
            return self._file.getall((*self._keywords, keyword))

        @override
        def __setitem__(
            self,
            keyword: str,
            data: DataLike | SubDictLike,
        ) -> None:
            self._file[(*self._keywords, keyword)] = data

        @override
        def add(self, keyword: str, data: DataLike | SubDictLike) -> None:
            self._file.add((*self._keywords, keyword), data)

        @override
        @with_default
        def popone(self, keyword: str) -> Data | FoamFile.SubDict:
            return self._file.popone((*self._keywords, keyword))

        @override
        def __delitem__(self, keyword: str) -> None:
            del self._file[(*self._keywords, keyword)]

        @override
        def __iter__(self) -> Iterator[str]:
            for k in self._file._iter(self._keywords):
                assert k is not None
                yield k

        @override
        def __contains__(self, keyword: object) -> bool:
            return (*self._keywords, keyword) in self._file

        @override
        def __len__(self) -> int:
            return len(list(iter(self)))

        @override
        def keys(self) -> FoamFile.SubDict.KeysView:
            return FoamFile.SubDict.KeysView(self)

        @override
        def values(self) -> FoamFile.SubDict.ValuesView:
            return FoamFile.SubDict.ValuesView(self)

        @override
        def items(self) -> FoamFile.SubDict.ItemsView:
            return FoamFile.SubDict.ItemsView(self)

        @override
        def update(
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike]
            | Iterable[tuple[str, DataLike | SubDictLike]] = (),
            /,
            **kwargs: DataLike | SubDictLike,
        ) -> None:
            with self._file:
                super().update(other, **kwargs)

        @override
        def extend(
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike]
            | Iterable[tuple[str, DataLike | SubDictLike]] = (),
            /,
            **kwargs: DataLike | SubDictLike,
        ) -> None:
            with self._file:
                super().extend(other, **kwargs)

        @override
        def merge(
            self,
            other: SupportsKeysAndGetItem[str, DataLike | SubDictLike]
            | Iterable[tuple[str, DataLike | SubDictLike]] = (),
            /,
            **kwargs: DataLike | SubDictLike,
        ) -> None:
            with self._file:
                super().merge(other, **kwargs)

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

            return ret

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

    @override
    @with_default
    def getall(
        self,
        keywords: str | tuple[str, ...] | None,
    ) -> Collection[Data | StandaloneData | FoamFile.SubDict]:
        if keywords is None:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        parsed = self._get_parsed()

        ret = parsed.getall(keywords)

        return [
            FoamFile.SubDict(self, keywords) if v is ... else deepcopy(v) for v in ret
        ]

    def _normalize_and_validate_keywords(
        self, keywords: str | tuple[str, ...] | None
    ) -> tuple[str, ...]:
        """Normalize keywords to tuple format and validate them."""
        if keywords is None:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        if keywords and not isinstance(normalize(keywords[-1], bool_ok=False), str):
            msg = f"Invalid keyword type: {keywords[-1]} (type {type(keywords[-1])})"
            raise TypeError(msg)

        return keywords

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
            self.object_ = (
                self.path.stem if self.path.suffix == ".gz" else self.path.name
            )

    def _update_class_for_field_if_needed(
        self,
        keywords: tuple[str, ...],
        data: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None:
        """Update class field to appropriate field type if this is a field entry."""
        if (
            keywords == ("internalField",)
            or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and (
                    keywords[2] == "value"
                    or keywords[2] == "gradient"
                    or keywords[2].endswith(("Value", "Gradient"))
                )
            )
        ) and self.class_ == "dictionary":
            try:
                tensor_kind = _tensor_kind_for_field(data)
            except ValueError:
                pass
            else:
                self.class_ = "vol" + tensor_kind[0].upper() + tensor_kind[1:] + "Field"

    def _calculate_spacing(
        self,
        keywords: tuple[str, ...],
        start: int,
        end: int,
        operation: Literal["put", "add"],
    ) -> tuple[bytes, bytes]:
        """Calculate before/after spacing for entry operations."""
        parsed = self._get_parsed(missing_ok=True)

        # For setitem operations, check if this is an update to an existing entry
        # and preserve existing spacing for sub-dictionary entries
        if operation == "put":
            is_update = keywords in parsed
            if is_update and len(keywords) > 1:
                # For existing sub-dictionary entries, preserve existing formatting
                existing_content = parsed.contents[start:end]
                if existing_content.startswith(b"\n"):
                    before = b""  # Preserve existing leading newlines
                elif parsed.contents[:start].endswith(b"\n"):
                    before = b""  # Already have a newline before
                else:
                    before = b"\n"  # Need to add a newline
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

        # Calculate after spacing (same for both operations)
        if not parsed.contents[end:].strip() or parsed.contents[end:].startswith(b"}"):
            after = b"\n" + b"    " * (len(keywords) - 2)
        else:
            after = b""

        return before, after

    def _perform_entry_operation(
        self,
        keywords: str | tuple[str, ...] | None,
        data: DataLike | StandaloneDataLike | SubDictLike,
        operation: Literal["put", "add"],
    ) -> None:
        """Shared method for performing entry operations (setitem and add)."""
        keywords = self._normalize_and_validate_keywords(keywords)

        with self:
            self._write_header_if_needed(keywords)
            self._update_class_for_field_if_needed(keywords, data)

            parsed = self._get_parsed(missing_ok=True)
            start, end = parsed.entry_location(keywords, add=(operation == "add"))
            before, after = self._calculate_spacing(keywords, start, end, operation)
            self._process_data_entry(keywords, data, before, after, operation)

    def _process_data_entry(
        self,
        keywords: tuple[str, ...],
        data: DataLike | StandaloneDataLike | SubDictLike,
        before: bytes,
        after: bytes,
        operation: Literal["put", "add"],
    ) -> None:
        """Process and write a data entry using either put or add operation."""
        parsed = self._get_parsed(missing_ok=True)
        indentation = b"    " * (len(keywords) - 1)

        if isinstance(data, Mapping):
            if not keywords:
                msg = "Cannot set a mapping at the root level of a FoamFile\nUse update(), extend(), or merge() instead."
                raise ValueError(msg)

            keyword = normalize(keywords[-1], bool_ok=False)

            if not isinstance(keyword, str):
                msg = (
                    f"Invalid keyword type: {keywords[-1]} (type {type(keywords[-1])})"
                )
                raise TypeError(msg)

            if keyword.startswith("#"):
                msg = (
                    f"Cannot set a directive as the keyword for a dictionary: {keyword}"
                )
                raise ValueError(msg)

            data = normalize(data, keywords=keywords)

            content = (
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

            if operation == "add" and keywords in parsed:
                raise KeyError(keywords)

            parsed.put(keywords, ..., content)
            for k, v in data.items():
                self[(*keywords, k)] = v

        elif keywords:
            header = self.get("FoamFile", None)
            assert header is None or isinstance(header, FoamFile.SubDict)
            val = dumps(data, keywords=keywords, header=header)

            content = (
                before
                + indentation
                + dumps(normalize(keywords[-1], bool_ok=False))
                + ((b" " + val) if val else b"")
                + (b";" if not keywords[-1].startswith("#") else b"")
                + after
            )

            if operation == "put":
                parsed.put(keywords, normalize(data, keywords=keywords), content)
            else:  # operation == "add"
                if keywords in parsed and not keywords[-1].startswith("#"):
                    raise KeyError(keywords)
                parsed.add(keywords, normalize(data, keywords=keywords), content)

        else:
            if operation == "add" and () in parsed:
                raise KeyError(())

            header = self.get("FoamFile", None)
            assert header is None or isinstance(header, FoamFile.SubDict)

            content = before + dumps(data, keywords=(), header=header) + after

            parsed.put((), normalize(data, keywords=keywords), content)

    @overload
    def __setitem__(
        self, keywords: None | tuple[()], data: StandaloneDataLike
    ) -> None: ...

    @overload
    def __setitem__(self, keywords: str, data: DataLike | SubDictLike) -> None: ...

    @overload
    def __setitem__(
        self,
        keywords: tuple[str, ...],
        data: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None: ...

    @override
    def __setitem__(
        self,
        keywords: str | tuple[str, ...] | None,
        data: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None:
        self._perform_entry_operation(keywords, data, "put")

    @override
    def add(
        self,
        keywords: str | tuple[str, ...] | None,
        data: Data | StandaloneData | SubDictLike,
    ) -> None:
        self._perform_entry_operation(keywords, data, "add")

    @with_default
    @override
    def popone(
        self, keywords: str | tuple[str, ...] | None
    ) -> Data | StandaloneData | FoamFile.SubDict:
        if keywords is None:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        with self:
            return self._get_parsed().popone(keywords)

    @overload
    def _iter(
        self, keywords: tuple[str, ...], *, include_header: bool = False
    ) -> Iterator[str]: ...

    @overload
    def _iter(
        self, keywords: tuple[()] = (), *, include_header: bool = False
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
        if keywords is None:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        return keywords in self._get_parsed()

    @override
    def __len__(self) -> int:
        """Return the number of top-level keywords in the FoamFile (excluding the FoamFile header if present)."""
        return len(list(iter(self)))

    @override
    def keys(
        self,
        *,
        include_header: bool = False,
    ) -> FoamFile.KeysView:
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
    ) -> FoamFile.ValuesView:
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
    ) -> FoamFile.ItemsView:
        """
        Return a collection of the items (keyword-value pairs) in the FoamFile.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        return FoamFile.ItemsView(self, include_header=include_header)

    @override
    def update(
        self,
        other: SupportsKeysAndGetItem[
            str | tuple[str, ...] | None, DataLike | StandaloneDataLike | SubDictLike
        ]
        | Iterable[
            tuple[
                str | tuple[str, ...] | None,
                DataLike | StandaloneDataLike | SubDictLike,
            ]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None:
        with self:
            super().update(other, **kwargs)

    @override
    def extend(
        self,
        other: SupportsKeysAndGetItem[
            str | tuple[str, ...] | None, DataLike | StandaloneDataLike | SubDictLike
        ]
        | Iterable[
            tuple[
                str | tuple[str, ...] | None,
                DataLike | StandaloneDataLike | SubDictLike,
            ]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None:
        with self:
            super().extend(other, **kwargs)

    @override
    def merge(
        self,
        other: SupportsKeysAndGetItem[
            str | tuple[str, ...] | None, DataLike | StandaloneDataLike | SubDictLike
        ]
        | Iterable[
            tuple[
                str | tuple[str, ...] | None,
                DataLike | StandaloneDataLike | SubDictLike,
            ]
        ] = (),
        /,
        **kwargs: DataLike | StandaloneDataLike | SubDictLike,
    ) -> None:
        with self:
            super().merge(other, **kwargs)

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

    def __fspath__(self) -> str:
        return str(self.path)

    def as_dict(self, *, include_header: bool = False) -> File:
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
        s: bytes | str,
        *,
        include_header: bool = False,
    ) -> File | StandaloneData:
        """
        Standalone deserializing function.

        Deserialize the OpenFOAM FoamFile format to Python objects.

        :param s: The string to deserialize. This can be a dictionary, list, or any
            other object that can be serialized to the OpenFOAM format.
        :param include_header: Whether to include the "FoamFile" header in the output.
            If `True`, the header will be included if it is present in the input object.
        """
        file = Parsed(s).as_dict()

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
        file: FileLike | StandaloneDataLike, *, ensure_header: bool = True
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
        header: SubDictLike | None
        if isinstance(file, Mapping):
            h = file.get("FoamFile", None)
            assert h is None or isinstance(h, Mapping)
            header = h

            entries: list[bytes] = []
            for k, v in file.items():
                if k is not None:
                    v = cast("Data | SubDict", v)
                    entries.append(
                        dumps(
                            (k, v),
                            keywords=(),
                            header=header,
                            tuple_is_keyword_entry=True,
                        )
                    )
                else:
                    assert not isinstance(v, Mapping)
                    entries.append(dumps(v, keywords=(), header=header))
            ret = b" ".join(entries)
        else:
            header = None
            ret = dumps(file, keywords=(), header=header)

        if header is None and ensure_header:
            class_ = "dictionary"
            if isinstance(file, Mapping) and "internalField" in file:
                try:
                    tensor_kind = _tensor_kind_for_field(file["internalField"])
                except (ValueError, TypeError):
                    pass
                else:
                    class_ = "vol" + tensor_kind[0].upper() + tensor_kind[1:] + "Field"

            header = {"version": 2.0, "format": "ascii", "class": class_}

            ret = (
                dumps(
                    ("FoamFile", header),
                    tuple_is_keyword_entry=True,
                )
                + b" "
                + ret
            )

        return ret


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
            self, keyword: str
        ) -> Collection[FoamFieldFile.BoundarySubDict | Data]:
            ret = super().getall(keyword)
            for r in ret:
                if isinstance(r, FoamFile.SubDict):
                    assert isinstance(r, FoamFieldFile.BoundarySubDict)
            return ret

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

    @override
    @with_default
    def getall(
        self, keywords: str | tuple[str, ...] | None
    ) -> Collection[Data | StandaloneData | FoamFieldFile.SubDict]:
        if keywords is None:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        ret = list(super().getall(keywords))

        if keywords[0] == "boundaryField":
            for i, r in enumerate(ret):
                if isinstance(r, FoamFile.SubDict):
                    if len(keywords) == 1:
                        ret[i] = FoamFieldFile.BoundariesSubDict(self, keywords)
                    elif len(keywords) == 2:
                        ret[i] = FoamFieldFile.BoundarySubDict(self, keywords)

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
    def boundary_field(self) -> FoamFieldFile.BoundariesSubDict:
        """Alias of ``self["boundaryField"]``."""
        ret = self["boundaryField"]
        if not isinstance(ret, FoamFieldFile.BoundariesSubDict):
            assert not isinstance(ret, FoamFile.SubDict)
            msg = "boundaryField is not a dictionary"
            raise TypeError(msg)
        return ret

    @boundary_field.setter
    def boundary_field(self, value: Mapping[str, SubDict]) -> None:
        self["boundaryField"] = value
