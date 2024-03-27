from pathlib import Path
from dataclasses import dataclass
from collections import namedtuple
from contextlib import suppress
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

from pyparsing import (
    Dict,
    Forward,
    Group,
    Keyword,
    LineEnd,
    Literal,
    Located,
    Opt,
    QuotedString,
    Word,
    c_style_comment,
    common,
    cpp_style_comment,
    printables,
    identchars,
    identbodychars,
)

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    numpy = False
else:
    numpy = True

from ._subprocesses import run_process, CalledProcessError


class _FoamDictionary(MutableMapping[str, Union["FoamFile.Value", "_FoamDictionary"]]):

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

    def __getitem__(self, key: str) -> Union["FoamFile.Value", "_FoamDictionary"]:
        contents = self._file.path.read_text()
        value = _DICTIONARY.parse_string(contents, parse_all=True).as_dict()

        for key in [*self._keywords, key]:
            value = value[key]

        if isinstance(value, dict):
            return _FoamDictionary(self._file, [*self._keywords, key])
        else:
            start, end = value
            return _VALUE.parse_string(contents[start:end], parse_all=True).as_list()[0]

    def _setitem(
        self,
        key: str,
        value: Any,
        *,
        assume_field: bool = False,
        assume_dimensions: bool = False,
    ) -> None:
        if isinstance(value, _FoamDictionary):
            value = value._cmd(["-value"])
        elif isinstance(value, Mapping):
            self._cmd(["-set", "{}"], key=key)
            subdict = self[key]
            print(subdict)
            assert isinstance(subdict, _FoamDictionary)
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
        value = _DICTIONARY.parse_file(self._file.path, parse_all=True).as_dict()

        for key in self._keywords:
            value = value[key]

        yield from value

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __repr__(self) -> str:
        return "FoamFile.Dictionary"


class FoamFile(_FoamDictionary):
    """An OpenFOAM dictionary file as a mutable mapping."""

    Dictionary = _FoamDictionary

    DimensionSet = namedtuple(
        "DimensionSet",
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
    class Dimensioned:
        value: Union[int, float, Sequence[Union[int, float]]] = 0
        dimensions: Union["FoamFile.DimensionSet", Sequence[Union[int, float]]] = ()
        name: Optional[str] = None

        def __post_init__(self) -> None:
            if not isinstance(self.dimensions, FoamFile.DimensionSet):
                self.dimensions = FoamFile.DimensionSet(*self.dimensions)

    Value = Union[str, int, float, bool, Dimensioned, DimensionSet, Sequence["Value"]]
    """
    A value that can be stored in an OpenFOAM dictionary.
    """

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

    class BoundariesDictionary(_FoamDictionary):
        def __getitem__(
            self, key: str
        ) -> Union["FoamFile.Value", "FoamFieldFile.BoundaryDictionary"]:
            ret = super().__getitem__(key)
            if isinstance(ret, _FoamDictionary):
                ret = FoamFieldFile.BoundaryDictionary(
                    self._file, [*self._keywords, key]
                )
            return ret

    class BoundaryDictionary(_FoamDictionary):
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

        def __repr__(self) -> str:
            return "FoamFieldFile.BoundaryDictionary"

    def __getitem__(self, key: str) -> Union[FoamFile.Value, _FoamDictionary]:
        ret = super().__getitem__(key)
        if key == "boundaryField" and isinstance(ret, _FoamDictionary):
            ret = FoamFieldFile.BoundariesDictionary(self, [key])
        return ret

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "internalField":
            self._setitem(key, value, assume_field=True)
        elif key == "dimensions":
            self._setitem(key, value, assume_dimensions=True)
        else:
            self._setitem(key, value)

    def __repr__(self) -> str:
        return "FoamFieldFile.BoundariesDictionary"

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
            assert not isinstance(ret, _FoamDictionary)
            raise TypeError("boundaryField is not a dictionary")
        return ret


_YES = Keyword("yes").set_parse_action(lambda: True)
_NO = Keyword("no").set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamFile.DimensionSet(*tks))
_TOKEN = QuotedString('"', unquote_results=False) | Word(
    identchars + "$", identbodychars
)
_ITEM = Forward()
_LIST = Opt(
    Literal("List") + Literal("<") + common.identifier + Literal(">")
).suppress() + (
    (
        Opt(common.integer).suppress()
        + Literal("(").suppress()
        + Group(_ITEM[...])
        + Literal(")").suppress()
    )
    | (
        common.integer + Literal("{").suppress() + _ITEM + Literal("}").suppress()
    ).set_parse_action(lambda tks: [tks[1]] * tks[0])
)
_FIELD = (Keyword("uniform").suppress() + _ITEM) | (
    Keyword("nonuniform").suppress() + _LIST
)
_DIMENSIONED = (Opt(common.identifier) + _DIMENSIONS + _ITEM).set_parse_action(
    lambda tks: FoamFile.Dimensioned(*reversed(tks.as_list()))
)
_ITEM <<= (
    _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _YES | _NO | _TOKEN
)

_TOKENS = (
    QuotedString('"', unquote_results=False)
    | Word(printables.replace(";", "").replace("{", "").replace("}", ""))
)[2, ...].set_parse_action(lambda tks: " ".join(tks))

_VALUE = (_ITEM ^ _TOKENS).ignore(c_style_comment).ignore(cpp_style_comment)


_UNPARSED_VALUE = (
    QuotedString('"', unquote_results=False)
    | Word(printables.replace(";", "").replace("{", "").replace("}", ""))
)[...]
_KEYWORD = QuotedString('"', unquote_results=False) | Word(
    identchars + "$(,.)", identbodychars + "$(,.)"
)
_DICTIONARY = Forward()
_ENTRY = _KEYWORD + (
    (
        Located(_UNPARSED_VALUE).set_parse_action(lambda tks: (tks[0], tks[2]))
        + Literal(";").suppress()
    )
    | (Literal("{").suppress() + _DICTIONARY + Literal("}").suppress())
)
_DICTIONARY <<= (
    Dict(Group(_ENTRY)[...])
    .set_parse_action(lambda tks: {} if not tks else tks)
    .ignore(c_style_comment)
    .ignore(cpp_style_comment)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore
)


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
        return f"({' '.join(serialize(v) for v in value)})"
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
    if isinstance(value, FoamFile.Dimensioned):
        if value.name is not None:
            return f"{value.name} {_serialize_dimensions(value.dimensions)} {serialize(value.value)}"
        else:
            return f"{_serialize_dimensions(value.dimensions)} {serialize(value.value)}"
    else:
        raise TypeError(f"Not a valid dimensioned value: {type(value)}")


def serialize(
    value: Any, *, assume_field: bool = False, assume_dimensions: bool = False
) -> str:
    if isinstance(value, FoamFile.DimensionSet) or assume_dimensions:
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
