from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence, Tuple, Union, cast

from ._base import FoamDictionaryBase
from ._parsing import entry_locn, parse
from ._serialization import serialize_value

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    pass


class FoamFile(
    FoamDictionaryBase,
    MutableMapping[str, Union["FoamFile.Value", "FoamFile.Dictionary"]],
):
    """An OpenFOAM dictionary file as a mutable mapping."""

    class Dictionary(
        FoamDictionaryBase,
        MutableMapping[str, Union["FoamFile.Value", "FoamFile.Dictionary"]],
    ):
        def __init__(self, _file: "FoamFile", _keywords: Sequence[str]) -> None:
            self._file = _file
            self._keywords = _keywords

        def __getitem__(
            self, keyword: str
        ) -> Union["FoamFile.Value", "FoamFile.Dictionary"]:
            return self._file[(*self._keywords, keyword)]

        def _setitem(
            self,
            keyword: str,
            value: Any,
            *,
            assume_field: bool = False,
            assume_dimensions: bool = False,
        ) -> None:
            self._file._setitem(
                (*self._keywords, keyword),
                value,
                assume_field=assume_field,
                assume_dimensions=assume_dimensions,
            )

        def __setitem__(self, keyword: str, value: Any) -> None:
            self._setitem(keyword, value)

        def __delitem__(self, keyword: str) -> None:
            del self._file[(*self._keywords, keyword)]

        def __iter__(self) -> Iterator[str]:
            return self._file._iter(tuple(self._keywords))

        def __len__(self) -> int:
            return len(list(iter(self)))

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({self._file}, {self._keywords})"

        def as_dict(self) -> FoamDictionaryBase._Dict:
            """
            Return a nested dict representation of the dictionary.
            """
            ret = self._file.as_dict()

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = v

            return ret

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).absolute()
        if self.path.is_dir():
            raise IsADirectoryError(self.path)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union["FoamFile.Value", "FoamFile.Dictionary"]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self.path.read_text()
        parsed = parse(contents)

        _, value, _ = parsed[keywords]

        if value is None:
            return FoamFile.Dictionary(self, keywords)
        else:
            return value

    def _setitem(
        self,
        keywords: Union[str, Tuple[str, ...]],
        value: Any,
        *,
        assume_field: bool = False,
        assume_dimensions: bool = False,
    ) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self.path.read_text()
        parsed = parse(contents)

        if isinstance(value, Mapping):
            if isinstance(value, FoamFile.Dictionary):
                value = value.as_dict()

            start, end = entry_locn(parsed, keywords)

            contents = f"{contents[:start]} {keywords[-1]} {{\n}}\n {contents[end:]}"
            self.path.write_text(contents)

            for k, v in value.items():
                self[(*keywords, k)] = v
        else:
            start, end = entry_locn(parsed, keywords)

            value = serialize_value(
                value, assume_field=assume_field, assume_dimensions=assume_dimensions
            )

            contents = f"{contents[:start]} {keywords[-1]} {value};\n {contents[end:]}"
            self.path.write_text(contents)

    def __setitem__(self, keywords: Union[str, Tuple[str, ...]], value: Any) -> None:
        self._setitem(keywords, value)

    def __delitem__(self, keywords: Union[str, Tuple[str, ...]]) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self.path.read_text()
        parsed = parse(contents)

        start, _, end = parsed[keywords]

        self.path.write_text(contents[:start] + contents[end:])

    def _iter(self, keywords: Union[str, Tuple[str, ...]] = ()) -> Iterator[str]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self.path.read_text()
        parsed = parse(contents)

        yield from (k[-1] for k in parsed if k[:-1] == keywords)

    def __iter__(self) -> Iterator[str]:
        return self._iter()

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def as_dict(self) -> FoamDictionaryBase._Dict:
        """
        Return a nested dict representation of the file.
        """
        contents = self.path.read_text()
        parsed = parse(contents)
        ret: FoamDictionaryBase._Dict = {}
        for keywords, (_, value, _) in parsed.items():

            r = ret
            for k in keywords[:-1]:
                assert isinstance(r, dict)
                v = r[k]
                assert isinstance(v, dict)
                r = v

            assert isinstance(r, dict)
            r[keywords[-1]] = {} if value is None else value

        return ret


class FoamFieldFile(FoamFile):
    """An OpenFOAM dictionary file representing a field as a mutable mapping."""

    class BoundariesDictionary(FoamFile.Dictionary):
        def __getitem__(self, keyword: str) -> "FoamFieldFile.BoundaryDictionary":
            return cast(FoamFieldFile.BoundaryDictionary, super().__getitem__(keyword))

    class BoundaryDictionary(FoamFile.Dictionary):
        """An OpenFOAM dictionary representing a boundary condition as a mutable mapping."""

        def __setitem__(self, key: str, value: Any) -> None:
            if key == "value":
                self._setitem(key, value, assume_field=True)
            else:
                self._setitem(key, value)

        @property
        def type(self) -> str:
            """
            Alias of `self["type"]`.
            """
            ret = self["type"]
            if not isinstance(ret, str):
                raise TypeError("type is not a string")
            return ret

        @type.setter
        def type(self, value: str) -> None:
            self["type"] = value

        @property
        def value(
            self,
        ) -> Union[
            int,
            float,
            Sequence[Union[int, float, Sequence[Union[int, float]]]],
            "NDArray[np.generic]",
        ]:
            """
            Alias of `self["value"]`.
            """
            ret = self["value"]
            if not isinstance(ret, (int, float, Sequence)):
                raise TypeError("value is not a field")
            return cast(Union[int, float, Sequence[Union[int, float]]], ret)

        @value.setter
        def value(
            self,
            value: Union[
                int,
                float,
                Sequence[Union[int, float, Sequence[Union[int, float]]]],
                "NDArray[np.generic]",
            ],
        ) -> None:
            self["value"] = value

        @value.deleter
        def value(self) -> None:
            del self["value"]

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union[FoamFile.Value, FoamFile.Dictionary]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        ret = super().__getitem__(keywords)
        if keywords[0] == "boundaryField" and isinstance(ret, FoamFile.Dictionary):
            if len(keywords) == 1:
                ret = FoamFieldFile.BoundariesDictionary(self, keywords)
            elif len(keywords) == 2:
                ret = FoamFieldFile.BoundaryDictionary(self, keywords)
        return ret

    def __setitem__(self, keywords: Union[str, Tuple[str, ...]], value: Any) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        if keywords == ("internalField",):
            self._setitem(keywords, value, assume_field=True)
        elif keywords == ("dimensions",):
            self._setitem(keywords, value, assume_dimensions=True)
        else:
            self._setitem(keywords, value)

    @property
    def dimensions(self) -> FoamFile.DimensionSet:
        """
        Alias of `self["dimensions"]`.
        """
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
        "NDArray[np.generic]",
    ]:
        """
        Alias of `self["internalField"]`.
        """
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
            "NDArray[np.generic]",
        ],
    ) -> None:
        self["internalField"] = value

    @property
    def boundary_field(self) -> "FoamFieldFile.BoundariesDictionary":
        """
        Alias of `self["boundaryField"]`.
        """
        ret = self["boundaryField"]
        if not isinstance(ret, FoamFieldFile.BoundariesDictionary):
            assert not isinstance(ret, FoamFile.Dictionary)
            raise TypeError("boundaryField is not a dictionary")
        return ret
