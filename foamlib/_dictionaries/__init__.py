from pathlib import Path
from typing import (
    Any,
    Union,
    Sequence,
    Iterator,
    Optional,
    Mapping,
    MutableMapping,
    cast,
)
from ._values import FoamDimensionSet, FoamDimensioned, FoamValue
from ._parsing import DICTIONARY, VALUE
from ._serialization import serialize
from .._subprocesses import run_process, CalledProcessError

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    pass


__all__ = [
    "FoamDictionary",
    "FoamBoundaryDictionary",
    "FoamBoundariesDictionary",
    "FoamFile",
    "FoamFieldFile",
    "FoamDimensionSet",
    "FoamDimensioned",
    "FoamValue",
]


class FoamDictionary(MutableMapping[str, Union[FoamValue, "FoamDictionary"]]):
    Value = FoamValue  # for backwards compatibility

    def __init__(self, _file: "FoamFile", _keywords: Sequence[str]) -> None:
        self._file = _file
        self._keywords = _keywords

    def _cmd(self, args: Sequence[str], *, key: Optional[str] = None) -> str:
        keywords = self._keywords

        if key is not None:
            keywords = [*self._keywords, key]

        if keywords:
            args = ["-entry", "/".join(keywords), *args]

        try:
            return (
                run_process(
                    ["foamDictionary", *args, "-precision", "15", self._file.path],
                )
                .stdout.decode()
                .strip()
            )
        except CalledProcessError as e:
            stderr = e.stderr.decode()
            if "Cannot find entry" in stderr:
                raise KeyError(key) from None
            else:
                raise RuntimeError(
                    f"{e.cmd} failed with return code {e.returncode}\n{e.stderr.decode()}"
                ) from None

    def __getitem__(self, key: str) -> Union[FoamValue, "FoamDictionary"]:
        contents = self._file.path.read_text()
        value = DICTIONARY.parse_string(contents, parse_all=True).as_dict()

        for key in [*self._keywords, key]:
            value = value[key]

        if isinstance(value, dict):
            return FoamDictionary(self._file, [*self._keywords, key])
        else:
            start, end = value
            return VALUE.parse_string(contents[start:end], parse_all=True).as_list()[0]

    def _setitem(
        self,
        key: str,
        value: Any,
        *,
        assume_field: bool = False,
        assume_dimensions: bool = False,
    ) -> None:
        if isinstance(value, FoamDictionary):
            value = value._cmd(["-value"])
        elif isinstance(value, Mapping):
            self._cmd(["-set", "{}"], key=key)
            subdict = self[key]
            print(subdict)
            assert isinstance(subdict, FoamDictionary)
            for k, v in value.items():
                subdict[k] = v
            return
        else:
            value = serialize(
                value, assume_field=assume_field, assume_dimensions=assume_dimensions
            )

        if len(value) < 1000:
            self._cmd(["-set", value], key=key)
        else:
            self._cmd(["-set", "_foamlib_value_"], key=key)
            contents = self._file.path.read_text()
            contents = contents.replace("_foamlib_value_", value, 1)
            self._file.path.write_text(contents)

    def __setitem__(self, key: str, value: Any) -> None:
        self._setitem(key, value)

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._cmd(["-remove"], key=key)

    def __iter__(self) -> Iterator[str]:
        value = DICTIONARY.parse_file(self._file.path, parse_all=True).as_dict()

        for key in self._keywords:
            value = value[key]

        yield from value

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __repr__(self) -> str:
        return type(self).__name__


class FoamBoundaryDictionary(FoamDictionary):
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


class FoamBoundariesDictionary(FoamDictionary):
    def __getitem__(self, key: str) -> Union[FoamValue, FoamBoundaryDictionary]:
        ret = super().__getitem__(key)
        if isinstance(ret, FoamDictionary):
            ret = FoamBoundaryDictionary(self._file, [*self._keywords, key])
        return ret


class FoamFile(FoamDictionary):
    """An OpenFOAM dictionary file as a mutable mapping."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(self, [])
        self.path = Path(path).absolute()
        if self.path.is_dir():
            raise IsADirectoryError(self.path)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"


class FoamFieldFile(FoamFile):
    """An OpenFOAM dictionary file representing a field as a mutable mapping."""

    def __getitem__(self, key: str) -> Union[FoamValue, FoamDictionary]:
        ret = super().__getitem__(key)
        if key == "boundaryField" and isinstance(ret, FoamDictionary):
            ret = FoamBoundariesDictionary(self, [key])
        return ret

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "internalField":
            self._setitem(key, value, assume_field=True)
        elif key == "dimensions":
            self._setitem(key, value, assume_dimensions=True)
        else:
            self._setitem(key, value)

    @property
    def dimensions(self) -> FoamDimensionSet:
        """
        Alias of `self["dimensions"]`.
        """
        ret = self["dimensions"]
        if not isinstance(ret, FoamDimensionSet):
            raise TypeError("dimensions is not a DimensionSet")
        return ret

    @dimensions.setter
    def dimensions(
        self, value: Union[FoamDimensionSet, Sequence[Union[int, float]]]
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
    def boundary_field(self) -> FoamBoundariesDictionary:
        """
        Alias of `self["boundaryField"]`.
        """
        ret = self["boundaryField"]
        if not isinstance(ret, FoamBoundariesDictionary):
            assert not isinstance(ret, FoamDictionary)
            raise TypeError("boundaryField is not a dictionary")
        return ret
