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
from collections import namedtuple
from dataclasses import dataclass
from contextlib import suppress

from ._subprocesses import run_process, CalledProcessError

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    numpy = False
else:
    numpy = True

from pyparsing import (
    Forward,
    Group,
    Keyword,
    Literal,
    Opt,
    ParseException,
    common,
)

FoamDimensionSet = namedtuple(
    "FoamDimensionSet",
    [
        "mass",
        "length",
        "time",
        "temperature",
        "moles",
        "current",
        "luminous_intensity",
    ],
    defaults=(0, 0, 0, 0, 0, 0, 0),
)


@dataclass
class FoamDimensioned:
    name: Optional[str] = None
    dimensions: Union[FoamDimensionSet, Sequence[Union[int, float]]] = (
        FoamDimensionSet()
    )
    value: Union[int, float, Sequence[Union[int, float]]] = 0

    def __post_init__(self) -> None:
        if self.name is not None and not isinstance(self.name, str) and self.value == 0:
            self.value = self.name
            self.name = None

        if not isinstance(self.dimensions, FoamDimensionSet):
            self.dimensions = FoamDimensionSet(*self.dimensions)


FoamValue = Union[
    str, int, float, bool, FoamDimensioned, FoamDimensionSet, Sequence["FoamValue"]
]
"""
A value that can be stored in an OpenFOAM dictionary.
"""

_YES = Keyword("yes").set_parse_action(lambda s, loc, tks: True)
_NO = Keyword("no").set_parse_action(lambda s, loc, tks: False)
_VALUE = Forward()
_LIST = Opt(
    Literal("List") + Literal("<") + common.identifier + Literal(">")
).suppress() + (
    (
        Opt(common.integer).suppress()
        + Literal("(").suppress()
        + Group(_VALUE[...])
        + Literal(")").suppress()
    )
    | (
        common.integer + Literal("{").suppress() + _VALUE + Literal("}").suppress()
    ).set_parse_action(lambda s, loc, tks: [tks[1]] * tks[0])
)
_FIELD = (Keyword("uniform").suppress() + _VALUE) | (
    Keyword("nonuniform").suppress() + _LIST
)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda s, loc, tks: FoamDimensionSet(*tks))
_DIMENSIONED = (common.identifier + _DIMENSIONS + _VALUE).set_parse_action(
    lambda s, loc, tks: FoamDimensioned(tks[0], tks[1], tks[2].as_list())
)

_VALUE << (_FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _YES | _NO)


def _parse(value: str) -> FoamValue:
    try:
        return cast(FoamValue, _VALUE.parse_string(value, parse_all=True).as_list()[0])
    except ParseException:
        return value


def _serialize_bool(value: Any) -> str:
    if value is True:
        return "yes"
    elif value is False:
        return "no"
    else:
        raise TypeError(f"Not a bool: {type(value)}")


def _is_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        or numpy
        and isinstance(value, np.ndarray)
    )


def _serialize_list(value: Any) -> str:
    if _is_sequence(value):
        return f"({' '.join(_serialize(v) for v in value)})"
    else:
        raise TypeError(f"Not a valid sequence: {type(value)}")


def _serialize_field(value: Any) -> str:
    if _is_sequence(value):
        try:
            s = _serialize_list(value)
        except TypeError:
            raise TypeError(f"Not a valid field: {type(value)}") from None
        else:
            if len(value) < 10:
                return f"uniform {s}"
            else:
                if isinstance(value[0], (int, float)):
                    kind = "scalar"
                elif len(value[0]) == 3:
                    kind = "vector"
                elif len(value[0]) == 6:
                    kind = "symmTensor"
                elif len(value[0]) == 9:
                    kind = "tensor"
                else:
                    raise TypeError(
                        f"Unsupported sequence length for field: {len(value[0])}"
                    )
                return f"nonuniform List<{kind}> {len(value)}{s}"
    else:
        return f"uniform {value}"


def _serialize_dimensions(value: Any) -> str:
    if _is_sequence(value) and len(value) == 7:
        return f"[{' '.join(str(v) for v in value)}]"
    else:
        raise TypeError(f"Not a valid dimension set: {type(value)}")


def _serialize_dimensioned(value: Any) -> str:
    if isinstance(value, FoamDimensioned):
        return f"{value.name or 'unnamed'} {_serialize_dimensions(value.dimensions)} {_serialize(value.value)}"
    else:
        raise TypeError(f"Not a valid dimensioned value: {type(value)}")


def _serialize(
    value: Any, *, assume_field: bool = False, assume_dimensions: bool = False
) -> str:
    if isinstance(value, FoamDimensionSet) or assume_dimensions:
        with suppress(TypeError):
            return _serialize_dimensions(value)

    if assume_field:
        with suppress(TypeError):
            return _serialize_field(value)

    with suppress(TypeError):
        return _serialize_dimensioned(value)

    with suppress(TypeError):
        return _serialize_list(value)

    with suppress(TypeError):
        return _serialize_bool(value)

    return str(value)


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
        value = self._cmd(["-value"], key=key)

        if value.startswith("{"):
            assert value.endswith("}")
            return FoamDictionary(self._file, [*self._keywords, key])
        else:
            return _parse(value)

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
            assert isinstance(subdict, FoamDictionary)
            for k, v in value.items():
                subdict[k] = v
            return
        else:
            value = _serialize(
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
        for key in self._cmd(["-keywords"]).splitlines():
            if not key.startswith('"'):
                yield key

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
        NDArray[np.generic],
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
        NDArray[np.generic],
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
