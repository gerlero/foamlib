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

from ._base import FoamFileBase
from ._io import FoamFileIO
from ._serialization import Kind, dumps
from ._util import is_sequence


class FoamFile(
    FoamFileBase,
    MutableMapping[
        Optional[Union[str, Tuple[str, ...]]],
        FoamFileBase._MutableData,
    ],
    FoamFileIO,
):
    """
    An OpenFOAM data file.

    Use as a mutable mapping (i.e., like a dict) to access and modify entries.

    Use as a context manager to make multiple changes to the file while saving all changes only once at the end.
    """

    class SubDict(
        MutableMapping[str, FoamFileBase._MutableData],
    ):
        """An OpenFOAM dictionary within a file as a mutable mapping."""

        def __init__(self, _file: FoamFile, _keywords: tuple[str, ...]) -> None:
            self._file = _file
            self._keywords = _keywords

        def __getitem__(
            self, keyword: str
        ) -> FoamFileBase._DataEntry | FoamFile.SubDict:
            return self._file[(*self._keywords, keyword)]

        def __setitem__(
            self,
            keyword: str,
            data: FoamFileBase.Data,
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

        def as_dict(self) -> FoamFileBase._Dict:
            """Return a nested dict representation of the dictionary."""
            ret = self._file.as_dict(include_header=True)

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = cast(FoamFileBase._File, v)

            return cast(FoamFileBase._Dict, ret)

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
        return cast(Literal["ascii", "binary"], ret)

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
    ) -> FoamFileBase._DataEntry | FoamFile.SubDict:
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
        self, keywords: str | tuple[str, ...] | None, data: FoamFileBase.Data
    ) -> None:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

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
                kind = Kind.BINARY_FIELD if self.format == "binary" else Kind.FIELD
            elif keywords == ("dimensions",):
                kind = Kind.DIMENSIONS

            if (
                kind in (Kind.FIELD, Kind.BINARY_FIELD)
            ) and self.class_ == "dictionary":
                if isinstance(data, (int, float)):
                    self.class_ = "volScalarField"

                elif is_sequence(data) and data:
                    if isinstance(data[0], (int, float)):
                        if len(data) == 3:
                            self.class_ = "volVectorField"
                        elif len(data) == 6:
                            self.class_ = "volSymmTensorField"
                        elif len(data) == 9:
                            self.class_ = "volTensorField"
                    elif (
                        is_sequence(data[0])
                        and data[0]
                        and isinstance(data[0][0], (int, float))
                    ):
                        if len(data[0]) == 3:
                            self.class_ = "volVectorField"
                        elif len(data[0]) == 6:
                            self.class_ = "volSymmTensorField"
                        elif len(data[0]) == 9:
                            self.class_ = "volTensorField"

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
                parsed.put(
                    keywords,
                    deepcopy(data),
                    before
                    + indentation
                    + dumps(keywords[-1])
                    + b" "
                    + dumps(data, kind=kind)
                    + b";"
                    + after,
                )

            else:
                parsed.put((), deepcopy(data), before + dumps(data, kind=kind) + after)

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

    def as_dict(self, *, include_header: bool = False) -> FoamFileBase._File:
        """
        Return a nested dict representation of the file.

        :param include_header: Whether to include the "FoamFile" header in the output.
        """
        d = self._get_parsed().as_dict()
        if not include_header:
            d.pop("FoamFile", None)
        return deepcopy(d)


class FoamFieldFile(FoamFile):
    """An OpenFOAM dictionary file representing a field as a mutable mapping."""

    class BoundariesSubDict(FoamFile.SubDict):
        def __getitem__(self, keyword: str) -> FoamFieldFile.BoundarySubDict:
            value = super().__getitem__(keyword)
            if not isinstance(value, FoamFieldFile.BoundarySubDict):
                assert not isinstance(value, FoamFile.SubDict)
                msg = f"boundary {keyword} is not a dictionary"
                raise TypeError(msg)
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
        ) -> FoamFile._Field:
            """Alias of `self["value"]`."""
            return cast(
                FoamFile._Field,
                self["value"],
            )

        @value.setter
        def value(
            self,
            value: FoamFile._Field,
        ) -> None:
            self["value"] = value

        @value.deleter
        def value(self) -> None:
            del self["value"]

    def __getitem__(
        self, keywords: str | tuple[str, ...] | None
    ) -> FoamFileBase._DataEntry | FoamFile.SubDict:
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
    def dimensions(self) -> FoamFile.DimensionSet | Sequence[float]:
        """Alias of `self["dimensions"]`."""
        ret = self["dimensions"]
        if not isinstance(ret, FoamFile.DimensionSet):
            msg = "dimensions is not a DimensionSet"
            raise TypeError(msg)
        return ret

    @dimensions.setter
    def dimensions(self, value: FoamFile.DimensionSet | Sequence[float]) -> None:
        self["dimensions"] = value

    @property
    def internal_field(
        self,
    ) -> FoamFile._Field:
        """Alias of `self["internalField"]`."""
        return cast(FoamFile._Field, self["internalField"])

    @internal_field.setter
    def internal_field(
        self,
        value: FoamFile._Field,
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
    def boundary_field(self, value: Mapping[str, FoamFile._Dict]) -> None:
        self["boundaryField"] = value
