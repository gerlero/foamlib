from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any, Optional, Tuple, Union, cast

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from ._io import FoamFileIO
from ._serialization import Kind, dumps, normalize
from ._types import (
    Data,
    Dict_,
    Dimensioned,
    DimensionSet,
    EntryLike,
    Field,
    FieldLike,
    File,
    MutableEntry,
)


class FoamFile(
    MutableMapping[
        Optional[Union[str, Tuple[str, ...]]],
        MutableEntry,
    ],
    FoamFileIO,
):
    """
    An OpenFOAM data file.

    `FoamFile` supports most OpenFOAM data and configuration files (i.e., files with a
    "FoamFile" header), including those with regular expressions and #-based directives.
    Notable exceptions are FoamFiles with #codeStreams and those multiple #-directives
    with the same name, which are currently not supported. Non-FoamFile output files are
    also not suppored by this class. Regular expressions and #-based directives can be
    accessed and modified, but they are not evaluated or expanded by this library.

    Use `FoamFile` as a mutable mapping (i.e., like a `dict`) to access and modify
    entries. When accessing a sub-dictionary, the returned value will be a
    `FoamFile.SubDict` object, that allows for further access and modification of nested
    dictionaries within the `FoamFile` in a single operation.

    If the `FoamFile` does not store a dictionary, the main stored value can be accessed
    and modified by passing `None` as the key (e.g., `file[None]`).

    You can also use the `FoamFile` as a context manager (i.e., within a `with` block)
    to make multiple changes to the file while saving any and all changes only once at
    the end.

    :param path: The path to the file. If the file does not exist, it will be created
        when the first change is made. However, if an attempt is made to access entries
        in a non-existent file, a `FileNotFoundError` will be raised.

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

    class SubDict(
        MutableMapping[str, MutableEntry],
    ):
        """
        An OpenFOAM sub-dictionary within a file.

        `FoamFile.SubDict` is a mutable mapping that allows for accessing and modifying
        nested dictionaries within a `FoamFile` in a single operation. It behaves like a
        `dict` and can be used to access and modify entries in the sub-dictionary.

        To obtain a `FoamFile.SubDict` object, access a sub-dictionary in a `FoamFile`
        object (e.g., `file["subDict"]`).

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

        def __init__(self, _file: FoamFile, _keywords: tuple[str, ...]) -> None:
            self._file = _file
            self._keywords = _keywords

        def __getitem__(self, keyword: str) -> Data | FoamFile.SubDict:
            return self._file[(*self._keywords, keyword)]

        def __setitem__(
            self,
            keyword: str,
            data: EntryLike,
        ) -> None:
            self._file[(*self._keywords, keyword)] = data

        def __delitem__(self, keyword: str) -> None:
            del self._file[(*self._keywords, keyword)]

        def __iter__(self) -> Iterator[str]:
            for k in self._file._iter(self._keywords):
                assert k is not None
                yield k

        def __contains__(self, keyword: object) -> bool:
            return (*self._keywords, keyword) in self._file

        def __len__(self) -> int:
            return len(list(iter(self)))

        def update(self, *args: Any, **kwargs: Any) -> None:
            with self._file:
                super().update(*args, **kwargs)

        def clear(self) -> None:
            with self._file:
                super().clear()

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self._file}', {self._keywords})"

        def as_dict(self) -> Dict_:
            """Return a nested dict representation of the sub-dictionary."""
            ret = self._file.as_dict(include_header=True)

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = cast("File", v)

            return cast("Dict_", ret)

    @property
    def version(self) -> float:
        """Alias of `self["FoamFile", "version"]`."""
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
        """Alias of `self["FoamFile", "format"]`."""
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
        """Alias of `self["FoamFile", "class"]`."""
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
        """Alias of `self["FoamFile", "location"]`."""
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
        """Alias of `self["FoamFile", "object"]`."""
        ret = self["FoamFile", "object"]
        if not isinstance(ret, str):
            msg = "object is not a string"
            raise TypeError(msg)
        return ret

    @object_.setter
    def object_(self, value: str) -> None:
        self["FoamFile", "object"] = value

    def __getitem__(
        self, keywords: str | tuple[str, ...] | None
    ) -> Data | FoamFile.SubDict:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        parsed = self._get_parsed()

        value = parsed[keywords]

        assert not isinstance(value, Mapping)

        if value is ...:
            return FoamFile.SubDict(self, keywords)
        return deepcopy(value)

    def __setitem__(
        self, keywords: str | tuple[str, ...] | None, data: EntryLike
    ) -> None:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        if keywords and not isinstance(normalize(keywords[-1], kind=Kind.KEYWORD), str):
            msg = f"Invalid keyword: {keywords[-1]}"
            raise ValueError(msg)

        with self:
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

            kind = Kind.DEFAULT
            if keywords == ("internalField",) or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and (
                    keywords[2] in ("value", "gradient")
                    or keywords[2].endswith("Value")
                    or keywords[2].endswith("Gradient")
                )
            ):
                kind = (
                    Kind.BINARY_FIELD if self.format == "binary" else Kind.ASCII_FIELD
                )
            elif keywords == ("dimensions",):
                kind = Kind.DIMENSIONS

            if (
                kind in (Kind.ASCII_FIELD, Kind.BINARY_FIELD)
            ) and self.class_ == "dictionary":
                try:
                    shape = np.shape(data)  # type: ignore [arg-type]
                except ValueError:
                    pass
                else:
                    if not shape:
                        self.class_ = "volScalarField"
                    elif shape == (3,):
                        self.class_ = "volVectorField"
                    elif shape == (6,):
                        self.class_ = "volSymmTensorField"
                    elif shape == (9,):
                        self.class_ = "volTensorField"
                    elif len(shape) == 1:
                        self.class_ = "volScalarField"
                    elif len(shape) == 2:
                        if shape[1] == 3:
                            self.class_ = "volVectorField"
                        elif shape[1] == 6:
                            self.class_ = "volSymmTensorField"
                        elif shape[1] == 9:
                            self.class_ = "volTensorField"

            if kind == Kind.ASCII_FIELD and self.class_.endswith("scalarField"):
                kind = Kind.SCALAR_ASCII_FIELD
            elif kind == Kind.BINARY_FIELD and self.class_.endswith("scalarField"):
                kind = Kind.SCALAR_BINARY_FIELD

            parsed = self._get_parsed(missing_ok=True)

            start, end = parsed.entry_location(keywords, missing_ok=True)

            if start and not parsed.contents[:start].endswith(b"\n\n"):
                if parsed.contents[:start].endswith(b"\n"):
                    before = b"\n" if len(keywords) <= 1 else b""
                else:
                    before = b"\n\n" if len(keywords) <= 1 else b"\n"
            else:
                before = b""

            if not parsed.contents[end:].strip() or parsed.contents[end:].startswith(
                b"}"
            ):
                after = b"\n" + b"    " * (len(keywords) - 2)
            else:
                after = b""

            indentation = b"    " * (len(keywords) - 1)

            if isinstance(data, Mapping):
                if isinstance(data, (FoamFile, FoamFile.SubDict)):
                    data = data.as_dict()

                parsed.put(
                    keywords,
                    ...,
                    before
                    + indentation
                    + dumps(keywords[-1])
                    + b"\n"
                    + indentation
                    + b"{\n"
                    + indentation
                    + b"}"
                    + after,
                )

                for k, v in data.items():
                    self[(*keywords, k)] = v

            elif keywords:
                val = dumps(data, kind=kind)
                parsed.put(
                    keywords,
                    normalize(data, kind=kind),
                    before
                    + indentation
                    + dumps(keywords[-1])
                    + ((b" " + val) if val else b"")
                    + (b";" if not keywords[-1].startswith("#") else b"")
                    + after,
                )

            else:
                parsed.put(
                    (),
                    normalize(data, kind=kind),
                    before + dumps(data, kind=kind) + after,
                )

    def __delitem__(self, keywords: str | tuple[str, ...] | None) -> None:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        with self:
            del self._get_parsed()[keywords]

    def _iter(self, keywords: tuple[str, ...] = ()) -> Iterator[str | None]:
        yield from (
            k[-1] if k else None for k in self._get_parsed() if k[:-1] == keywords
        )

    def __iter__(self) -> Iterator[str | None]:
        yield from (k for k in self._iter() if k != "FoamFile")

    def __contains__(self, keywords: object) -> bool:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        return keywords in self._get_parsed()

    def __len__(self) -> int:
        return len(list(iter(self)))

    def update(self, *args: Any, **kwargs: Any) -> None:
        with self:
            super().update(*args, **kwargs)

    def clear(self) -> None:
        with self:
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


class FoamFieldFile(FoamFile):
    """
    Subclass of `FoamFile` for representing OpenFOAM field files specifically.

    The difference between `FoamFieldFile` and `FoamFile` is that `FoamFieldFile` has
    the additional properties `dimensions`, `internal_field`, and `boundary_field` that
    are commonly found in OpenFOAM field files. Note that these are only a shorthand for
    accessing the corresponding entries in the file.

    See `FoamFile` for more information on how to read and edit OpenFOAM files.

    :param path: The path to the file. If the file does not exist, it will be created
        when the first change is made. However, if an attempt is made to access entries
        in a non-existent file, a `FileNotFoundError` will be raised.

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
        def __getitem__(self, keyword: str) -> FoamFieldFile.BoundarySubDict | Data:
            value = super().__getitem__(keyword)
            if isinstance(value, FoamFieldFile.SubDict):
                assert isinstance(value, FoamFieldFile.BoundarySubDict)
            return value

    class BoundarySubDict(FoamFile.SubDict):
        """An OpenFOAM dictionary representing a boundary condition as a mutable mapping."""

        @property
        def type(self) -> str:
            """Alias of `self["type"]`."""
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
            """Alias of `self["value"]`."""
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

    def __getitem__(
        self, keywords: str | tuple[str, ...] | None
    ) -> Data | FoamFile.SubDict:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        ret = super().__getitem__(keywords)
        if keywords[0] == "boundaryField" and isinstance(ret, FoamFile.SubDict):
            if len(keywords) == 1:
                ret = FoamFieldFile.BoundariesSubDict(self, keywords)
            elif len(keywords) == 2:
                ret = FoamFieldFile.BoundarySubDict(self, keywords)
        return ret

    @property
    def dimensions(self) -> DimensionSet | Sequence[float]:
        """Alias of `self["dimensions"]`."""
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
        """Alias of `self["internalField"]`."""
        return cast("Field", self["internalField"])

    @internal_field.setter
    def internal_field(
        self,
        value: FieldLike,
    ) -> None:
        self["internalField"] = value

    @property
    def boundary_field(self) -> FoamFieldFile.BoundariesSubDict:
        """Alias of `self["boundaryField"]`."""
        ret = self["boundaryField"]
        if not isinstance(ret, FoamFieldFile.BoundariesSubDict):
            assert not isinstance(ret, FoamFile.SubDict)
            msg = "boundaryField is not a dictionary"
            raise TypeError(msg)
        return ret

    @boundary_field.setter
    def boundary_field(self, value: Mapping[str, Dict_]) -> None:
        self["boundaryField"] = value
