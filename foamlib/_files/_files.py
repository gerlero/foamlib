import sys
from typing import Any, Optional, Tuple, Union, cast

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .._util import is_sequence
from ._base import FoamFileBase
from ._io import _FoamFileIO
from ._serialization import Kind, dumpb

try:
    import numpy as np
except ModuleNotFoundError:
    pass


class FoamFile(
    FoamFileBase,
    MutableMapping[
        Optional[Union[str, Tuple[str, ...]]],
        Union["FoamFile.Data", "FoamFile.SubDict"],
    ],
    _FoamFileIO,
):
    """
    An OpenFOAM data file.

    Use as a mutable mapping (i.e., like a dict) to access and modify entries.

    Use as a context manager to make multiple changes to the file while saving all changes only once at the end.
    """

    class SubDict(
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
            ret = self._file.as_dict()

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = cast(FoamFileBase._File, v)

            return cast(FoamFileBase._Dict, ret)

    def create(self, *, exist_ok: bool = False, parents: bool = False) -> Self:
        """
        Create the file.

        Parameters
        ----------
        exist_ok : bool, optional
            If False (the default), raise a FileExistsError if the file already exists.
            If True, do nothing if the file already exists.
        parents : bool, optional
            If True, also create parent directories as needed.
        """
        if self.path.exists():
            if not exist_ok:
                raise FileExistsError(self.path)
            else:
                return self

        if parents:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.path.touch()
        self._write_header()

        return self

    @property
    def version(self) -> float:
        """Alias of `self["FoamFile", "version"]`."""
        ret = self["FoamFile", "version"]
        if not isinstance(ret, float):
            raise TypeError("version is not a float")
        return ret

    @version.setter
    def version(self, value: float) -> None:
        self["FoamFile", "version"] = value

    @property
    def format(self) -> Literal["ascii", "binary"]:
        """Alias of `self["FoamFile", "format"]`."""
        ret = self["FoamFile", "format"]
        if not isinstance(ret, str):
            raise TypeError("format is not a string")
        if ret not in ("ascii", "binary"):
            raise ValueError("format is not 'ascii' or 'binary'")
        return cast(Literal["ascii", "binary"], ret)

    @format.setter
    def format(self, value: Literal["ascii", "binary"]) -> None:
        self["FoamFile", "format"] = value

    @property
    def class_(self) -> str:
        """Alias of `self["FoamFile", "class"]`."""
        ret = self["FoamFile", "class"]
        if not isinstance(ret, str):
            raise TypeError("class is not a string")
        return ret

    @class_.setter
    def class_(self, value: str) -> None:
        self["FoamFile", "class"] = value

    @property
    def location(self) -> str:
        """Alias of `self["FoamFile", "location"]`."""
        ret = self["FoamFile", "location"]
        if not isinstance(ret, str):
            raise TypeError("location is not a string")
        return ret

    @location.setter
    def location(self, value: str) -> None:
        self["FoamFile", "location"] = value

    @property
    def object_(self) -> str:
        """Alias of `self["FoamFile", "object"]`."""
        ret = self["FoamFile", "object"]
        if not isinstance(ret, str):
            raise TypeError("object is not a string")
        return ret

    @object_.setter
    def object_(self, value: str) -> None:
        self["FoamFile", "object"] = value

    def _write_header(self) -> None:
        assert "FoamFile" not in self
        assert not self

        self["FoamFile"] = {}
        self.version = 2.0
        self.format = "ascii"
        self.class_ = "dictionary"
        self.location = f'"{self.path.parent.name}"'
        self.object_ = self.path.name

    def __getitem__(
        self, keywords: Optional[Union[str, Tuple[str, ...]]]
    ) -> Union["FoamFile.Data", "FoamFile.SubDict"]:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        _, parsed = self._read()

        value = parsed[keywords]

        if value is ...:
            return FoamFile.SubDict(self, keywords)
        else:
            return value

    def __setitem__(
        self, keywords: Optional[Union[str, Tuple[str, ...]]], data: "FoamFile._SetData"
    ) -> None:
        with self:
            if not keywords:
                keywords = ()
            elif not isinstance(keywords, tuple):
                keywords = (keywords,)

            if not self and "FoamFile" not in self and keywords[0] != "FoamFile":
                self._write_header()

            kind = Kind.DEFAULT
            if keywords == ("internalField",) or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and keywords[2] == "value"
            ):
                kind = Kind.BINARY_FIELD if self.format == "binary" else Kind.FIELD
            elif keywords == ("dimensions",):
                kind = Kind.DIMENSIONS

            if (
                kind == Kind.FIELD or kind == Kind.BINARY_FIELD
            ) and self.class_ == "dictionary":
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

                self.class_ = class_
                self[keywords] = data

            else:
                contents, parsed = self._read()

                start, end = parsed.entry_location(keywords, missing_ok=True)

                before = contents[:start].rstrip() + b"\n"
                if len(keywords) <= 1:
                    before += b"\n"

                after = contents[end:]
                if after.startswith(b"}"):
                    after = b"    " * (len(keywords) - 2) + after
                if not after or after[:1] != b"\n":
                    after = b"\n" + after
                if len(keywords) <= 1 and len(after) > 1 and after[:2] != b"\n\n":
                    after = b"\n" + after

                indentation = b"    " * (len(keywords) - 1)

                if isinstance(data, Mapping):
                    if isinstance(data, (FoamFile, FoamFile.SubDict)):
                        data = data.as_dict()

                    self._write(
                        before
                        + indentation
                        + dumpb(keywords[-1])
                        + b"\n"
                        + indentation
                        + b"{\n"
                        + indentation
                        + b"}"
                        + after
                    )

                    for k, v in data.items():
                        self[(*keywords, k)] = v

                elif keywords:
                    self._write(
                        before
                        + indentation
                        + dumpb(keywords[-1])
                        + b" "
                        + dumpb(data, kind=kind)
                        + b";"
                        + after
                    )

                else:
                    self._write(before + dumpb(data, kind=kind) + after)

    def __delitem__(self, keywords: Optional[Union[str, Tuple[str, ...]]]) -> None:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents, parsed = self._read()

        start, end = parsed.entry_location(keywords)

        self._write(contents[:start] + contents[end:])

    def _iter(self, keywords: Tuple[str, ...] = ()) -> Iterator[Optional[str]]:
        _, parsed = self._read()

        yield from (
            k[-1] if k else None
            for k in parsed
            if k != ("FoamFile",) and k[:-1] == keywords
        )

    def __iter__(self) -> Iterator[Optional[str]]:
        return self._iter()

    def __contains__(self, keywords: object) -> bool:
        if not keywords:
            keywords = ()
        elif not isinstance(keywords, tuple):
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

    def as_dict(self) -> FoamFileBase._File:
        """Return a nested dict representation of the file."""
        _, parsed = self._read()
        d = parsed.as_dict()
        del d["FoamFile"]
        return d


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
        self, keywords: Optional[Union[str, Tuple[str, ...]]]
    ) -> Union[FoamFile.Data, FoamFile.SubDict]:
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
