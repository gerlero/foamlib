import sys
from typing import Any, Tuple, Union, cast

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

from .._util import is_sequence
from ._base import FoamDict
from ._io import FoamFileIO
from ._serialization import Kind, dumpb

try:
    import numpy as np
except ModuleNotFoundError:
    pass


class FoamFile(
    FoamDict,
    MutableMapping[
        Union[str, Tuple[str, ...]], Union["FoamFile.Data", "FoamFile.SubDict"]
    ],
    FoamFileIO,
):
    """
    An OpenFOAM data file.

    Use as a mutable mapping (i.e., like a dict) to access and modify entries.

    Use as a context manager to make multiple changes to the file while saving all changes only once at the end.
    """

    class SubDict(
        FoamDict,
        MutableMapping[str, Union["FoamFile.Data", "FoamFile.SubDict"]],
    ):
        """An OpenFOAM dictionary within a file as a mutable mapping."""

        def __init__(self, _file: "FoamFile", _keywords: Tuple[str, ...]) -> None:
            self._file = _file
            self._keywords = _keywords

        def __getitem__(
            self, keyword: str
        ) -> Union["FoamFile.Data", "FoamFile.SubDict"]:
            return self._file[(*self._keywords, keyword)]

        def __setitem__(
            self,
            keyword: str,
            data: "FoamFile._SetData",
        ) -> None:
            self._file[(*self._keywords, keyword)] = data

        def __delitem__(self, keyword: str) -> None:
            del self._file[(*self._keywords, keyword)]

        def __iter__(self) -> Iterator[str]:
            return self._file._iter(self._keywords)

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

        def as_dict(self) -> FoamDict._Dict:
            """Return a nested dict representation of the dictionary."""
            ret = self._file.as_dict()

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = v

            return ret

    class Header(SubDict):
        """The header of an OpenFOAM file."""

        def __init__(self, _file: "FoamFile") -> None:
            super().__init__(_file, ("FoamFile",))

        @property
        def version(self) -> float:
            """Alias of `self["version"]`."""
            ret = self["version"]
            if not isinstance(ret, float):
                raise TypeError("version is not a float")
            return ret

        @version.setter
        def version(self, data: float) -> None:
            self["version"] = data

        @property
        def format(self) -> str:
            """Alias of `self["format"]`."""
            ret = self["format"]
            if not isinstance(ret, str):
                raise TypeError("format is not a string")
            return ret

        @format.setter
        def format(self, data: str) -> None:
            self["format"] = data

        @property
        def class_(self) -> str:
            """Alias of `self["class"]`."""
            ret = self["class"]
            if not isinstance(ret, str):
                raise TypeError("class is not a string")
            return ret

        @class_.setter
        def class_(self, data: str) -> None:
            self["class"] = data

        @property
        def location(self) -> str:
            """Alias of `self["location"]`."""
            ret = self["location"]
            if not isinstance(ret, str):
                raise TypeError("location is not a string")
            return ret

        @location.setter
        def location(self, data: str) -> None:
            self["location"] = data

        @property
        def object(self) -> str:
            """Alias of `self["object"]`."""
            ret = self["object"]
            if not isinstance(ret, str):
                raise TypeError("object is not a string")
            return ret

    @property
    def header(self) -> Header:
        """Alias of `self["FoamFile"]`."""
        ret = self["FoamFile"]
        if not isinstance(ret, FoamFile.Header):
            assert not isinstance(ret, FoamFile.SubDict)
            raise TypeError("FoamFile is not a dictionary")
        return ret

    @header.setter
    def header(self, data: FoamDict._Dict) -> None:
        self["FoamFile"] = data

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union["FoamFile.Data", "FoamFile.SubDict"]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        _, parsed = self._read()

        value = parsed[keywords]

        if value is ...:
            if keywords == ("FoamFile",):
                return FoamFile.Header(self)
            else:
                return FoamFile.SubDict(self, keywords)
        else:
            return value

    def __setitem__(
        self, keywords: Union[str, Tuple[str, ...]], data: "FoamFile._SetData"
    ) -> None:
        with self:
            if not isinstance(keywords, tuple):
                keywords = (keywords,)

            if not self and keywords[0] != "FoamFile":
                self.header = {
                    "version": 2.0,
                    "format": "ascii",
                    "class": "dictionary",
                    "location": f'"{self.path.parent.name}"',
                    "object": self.path.name,
                }  # type: ignore [assignment]

            kind = Kind.DEFAULT
            if keywords == ("internalField",) or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and keywords[2] == "value"
            ):
                kind = (
                    Kind.BINARY_FIELD if self.header.format == "binary" else Kind.FIELD
                )
            elif keywords == ("dimensions",):
                kind = Kind.DIMENSIONS

            contents, parsed = self._read()

            if isinstance(data, Mapping):
                if isinstance(data, FoamDict):
                    data = data.as_dict()

                start, end = parsed.entry_location(keywords, missing_ok=True)

                self._write(
                    contents[:start]
                    + (b"\n" if contents[start - 1 : start] != b"\n" else b"")
                    + (b"  " * (len(keywords) - 1) if keywords else b"")
                    + dumpb({keywords[-1]: {}})
                    + b"\n"
                    + contents[end:]
                )

                for k, v in data.items():
                    self[(*keywords, k)] = v

            elif (
                kind == Kind.FIELD or kind == Kind.BINARY_FIELD
            ) and self.header.class_ == "dictionary":
                if not is_sequence(data):
                    class_ = "volScalarField"
                elif (len(data) == 3 and not is_sequence(data[0])) or len(data[0]) == 3:
                    class_ = "volVectorField"
                elif (len(data) == 6 and not is_sequence(data[0])) or len(data[0]) == 6:
                    class_ = "volSymmTensorField"
                elif (len(data) == 9 and not is_sequence(data[0])) or len(data[0]) == 9:
                    class_ = "volTensorField"
                else:
                    class_ = "volScalarField"

                self.header.class_ = class_
                self[keywords] = data

            else:
                start, end = parsed.entry_location(keywords, missing_ok=True)

                self._write(
                    contents[:start]
                    + (
                        b"\n"
                        if start > 0 and contents[start - 1 : start] != b"\n"
                        else b""
                    )
                    + (b"    " * (len(keywords) - 1) if keywords else b"")
                    + dumpb({keywords[-1]: data}, kind=kind)
                    + b"\n"
                    + contents[end:]
                )

    def __delitem__(self, keywords: Union[str, Tuple[str, ...]]) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents, parsed = self._read()

        start, end = parsed.entry_location(keywords)

        self._write(contents[:start] + contents[end:])

    def _iter(self, keywords: Union[str, Tuple[str, ...]] = ()) -> Iterator[str]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        _, parsed = self._read()

        yield from (k[-1] for k in parsed if k[:-1] == keywords)

    def __iter__(self) -> Iterator[str]:
        return self._iter()

    def __contains__(self, keywords: object) -> bool:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)
        _, parsed = self._read()
        return keywords in parsed

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

    def as_dict(self) -> FoamDict._Dict:
        """Return a nested dict representation of the file."""
        _, parsed = self._read()
        return parsed.as_dict()


class FoamFieldFile(FoamFile):
    """An OpenFOAM dictionary file representing a field as a mutable mapping."""

    class BoundariesSubDict(FoamFile.SubDict):
        def __getitem__(self, keyword: str) -> "FoamFieldFile.BoundarySubDict":
            value = super().__getitem__(keyword)
            if not isinstance(value, FoamFieldFile.BoundarySubDict):
                assert not isinstance(value, FoamFile.SubDict)
                raise TypeError(f"boundary {keyword} is not a dictionary")
            return value

    class BoundarySubDict(FoamFile.SubDict):
        """An OpenFOAM dictionary representing a boundary condition as a mutable mapping."""

        @property
        def type(self) -> str:
            """Alias of `self["type"]`."""
            ret = self["type"]
            if not isinstance(ret, str):
                raise TypeError("type is not a string")
            return ret

        @type.setter
        def type(self, data: str) -> None:
            self["type"] = data

        @property
        def value(
            self,
        ) -> Union[
            int,
            float,
            Sequence[Union[int, float, Sequence[Union[int, float]]]],
            "np.ndarray[Tuple[()], np.dtype[np.generic]]",
            "np.ndarray[Tuple[int], np.dtype[np.generic]]",
            "np.ndarray[Tuple[int, int], np.dtype[np.generic]]",
        ]:
            """Alias of `self["value"]`."""
            ret = self["value"]
            if not isinstance(ret, (int, float, Sequence)):
                raise TypeError("value is not a field")
            return cast(
                Union[
                    int, float, Sequence[Union[int, float, Sequence[Union[int, float]]]]
                ],
                ret,
            )

        @value.setter
        def value(
            self,
            value: Union[
                int,
                float,
                Sequence[Union[int, float, Sequence[Union[int, float]]]],
                "np.ndarray[Tuple[()], np.dtype[np.generic]]",
                "np.ndarray[Tuple[int], np.dtype[np.generic]]",
                "np.ndarray[Tuple[int, int], np.dtype[np.generic]]",
            ],
        ) -> None:
            self["value"] = value

        @value.deleter
        def value(self) -> None:
            del self["value"]

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union[FoamFile.Data, FoamFile.SubDict]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        ret = super().__getitem__(keywords)
        if keywords[0] == "boundaryField" and isinstance(ret, FoamFile.SubDict):
            if len(keywords) == 1:
                ret = FoamFieldFile.BoundariesSubDict(self, keywords)
            elif len(keywords) == 2:
                ret = FoamFieldFile.BoundarySubDict(self, keywords)
        return ret

    @property
    def dimensions(self) -> Union[FoamFile.DimensionSet, Sequence[Union[int, float]]]:
        """Alias of `self["dimensions"]`."""
        ret = self["dimensions"]
        if not isinstance(ret, FoamFile.DimensionSet):
            raise TypeError("dimensions is not a DimensionSet")
        return ret

    @dimensions.setter
    def dimensions(
        self, value: Union[FoamFile.DimensionSet, Sequence[Union[int, float]]]
    ) -> None:
        self["dimensions"] = value

    @property
    def internal_field(
        self,
    ) -> Union[
        int,
        float,
        Sequence[Union[int, float, Sequence[Union[int, float]]]],
        "np.ndarray[Tuple[()], np.dtype[np.generic]]",
        "np.ndarray[Tuple[int], np.dtype[np.generic]]",
        "np.ndarray[Tuple[int, int], np.dtype[np.generic]]",
    ]:
        """Alias of `self["internalField"]`."""
        ret = self["internalField"]
        if not isinstance(ret, (int, float, Sequence)):
            raise TypeError("internalField is not a field")
        return cast(Union[int, float, Sequence[Union[int, float]]], ret)

    @internal_field.setter
    def internal_field(
        self,
        value: Union[
            int,
            float,
            Sequence[Union[int, float, Sequence[Union[int, float]]]],
            "np.ndarray[Tuple[()], np.dtype[np.generic]]",
            "np.ndarray[Tuple[int], np.dtype[np.generic]]",
            "np.ndarray[Tuple[int, int], np.dtype[np.generic]]",
        ],
    ) -> None:
        self["internalField"] = value

    @property
    def boundary_field(self) -> "FoamFieldFile.BoundariesSubDict":
        """Alias of `self["boundaryField"]`."""
        ret = self["boundaryField"]
        if not isinstance(ret, FoamFieldFile.BoundariesSubDict):
            assert not isinstance(ret, FoamFile.SubDict)
            raise TypeError("boundaryField is not a dictionary")
        return ret
