from pathlib import Path
from dataclasses import dataclass
from contextlib import suppress
import typing
from typing import (
    Any,
    Union,
    Sequence,
    Iterator,
    Optional,
    Mapping,
    MutableMapping,
    NamedTuple,
    Tuple,
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
    ParseResults,
    ParserElement,
    QuotedString,
    Word,
    c_style_comment,
    common,
    cpp_style_comment,
    identbodychars,
    printables,
)

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    numpy = False
else:
    numpy = True


class _FoamDictionary(MutableMapping[str, Union["FoamFile.Value", "_FoamDictionary"]]):

    def __init__(self, _file: "FoamFile", _keywords: Sequence[str]) -> None:
        self._file = _file
        self._keywords = _keywords

    def __getitem__(self, keyword: str) -> Union["FoamFile.Value", "_FoamDictionary"]:
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
        return f"FoamFile.Dictionary({self._file}, {self._keywords})"

    _Dict = typing.Dict[str, Union["FoamFile.Value", "_Dict"]]

    def as_dict(self) -> _Dict:
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


class FoamFile(_FoamDictionary):
    """An OpenFOAM dictionary file as a mutable mapping."""

    Dictionary = _FoamDictionary

    class DimensionSet(NamedTuple):
        mass: Union[int, float] = 0
        length: Union[int, float] = 0
        time: Union[int, float] = 0
        temperature: Union[int, float] = 0
        moles: Union[int, float] = 0
        current: Union[int, float] = 0
        luminous_intensity: Union[int, float] = 0

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self) if v != 0)})"

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

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union["FoamFile.Value", "_FoamDictionary"]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self._file.path.read_text()
        parsed = _parse(contents)

        _, value, _ = parsed[keywords]

        if value is None:
            return _FoamDictionary(self._file, keywords)
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
        parsed = _parse(contents)

        if isinstance(value, Mapping):
            if isinstance(value, FoamFile.Dictionary):
                value = value.as_dict()

            start, end = _entry_locn(parsed, keywords)

            contents = f"{contents[:start]} {keywords[-1]} {{\n}}\n {contents[end:]}"
            self.path.write_text(contents)

            for k, v in value.items():
                self[(*keywords, k)] = v
        else:
            start, end = _entry_locn(parsed, keywords)

            value = _serialize_value(
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
        parsed = _parse(contents)

        start, _, end = parsed[keywords]

        self.path.write_text(contents[:start] + contents[end:])

    def _iter(self, keywords: Union[str, Tuple[str, ...]] = ()) -> Iterator[str]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents = self.path.read_text()
        parsed = _parse(contents)

        yield from (k[-1] for k in parsed if k[:-1] == keywords)

    def __iter__(self) -> Iterator[str]:
        return self._iter()

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def as_dict(self) -> _FoamDictionary._Dict:
        """
        Return a nested dict representation of the file.
        """
        contents = self.path.read_text()
        parsed = _parse(contents)
        ret: _FoamDictionary._Dict = {}
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

    class BoundariesDictionary(_FoamDictionary):
        def __getitem__(self, keyword: str) -> "FoamFieldFile.BoundaryDictionary":
            return cast(FoamFieldFile.BoundaryDictionary, super().__getitem__(keyword))

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({self._file}, {self._keywords})"

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
            return f"{type(self).__qualname__}({self._file}, {self._keywords})"

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union[FoamFile.Value, _FoamDictionary]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        ret = super().__getitem__(keywords)
        if keywords[0] == "boundaryField" and isinstance(ret, _FoamDictionary):
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
            assert not isinstance(ret, _FoamDictionary)
            raise TypeError("boundaryField is not a dictionary")
        return ret


_YES = Keyword("yes").set_parse_action(lambda: True)
_NO = Keyword("no").set_parse_action(lambda: False)
_DIMENSIONS = (
    Literal("[").suppress() + common.number * 7 + Literal("]").suppress()
).set_parse_action(lambda tks: FoamFile.DimensionSet(*tks))


def _list_of(elem: ParserElement) -> ParserElement:
    return Opt(
        Literal("List") + Literal("<") + common.identifier + Literal(">")
    ).suppress() + (
        (
            Opt(common.integer).suppress()
            + (
                Literal("(").suppress()
                + Group((elem)[...], aslist=True)
                + Literal(")").suppress()
            )
        )
        | (
            common.integer + Literal("{").suppress() + elem + Literal("}").suppress()
        ).set_parse_action(lambda tks: [[tks[1]] * tks[0]])
    )


_TENSOR = _list_of(common.number) | common.number
_IDENTIFIER = Word(identbodychars + "$", printables.replace(";", ""))
_DIMENSIONED = (Opt(_IDENTIFIER) + _DIMENSIONS + _TENSOR).set_parse_action(
    lambda tks: FoamFile.Dimensioned(*reversed(tks.as_list()))
)
_FIELD = (Keyword("uniform").suppress() + _TENSOR) | (
    Keyword("nonuniform").suppress() + _list_of(_TENSOR)
)
_TOKEN = QuotedString('"', unquote_results=False) | _IDENTIFIER
_ITEM = Forward()
_LIST = _list_of(_ITEM)
_ITEM <<= (
    _FIELD | _LIST | _DIMENSIONED | _DIMENSIONS | common.number | _YES | _NO | _TOKEN
)
_TOKENS = (
    QuotedString('"', unquote_results=False) | Word(printables.replace(";", ""))
)[2, ...].set_parse_action(lambda tks: " ".join(tks))

_VALUE = _ITEM ^ _TOKENS

_ENTRY = Forward()
_DICTIONARY = Dict(Group(_ENTRY)[...])
_ENTRY <<= Located(
    _TOKEN
    + (
        (Literal("{").suppress() + _DICTIONARY + Literal("}").suppress())
        | (Opt(_VALUE, default="") + Literal(";").suppress())
    )
)
_FILE = (
    _DICTIONARY.ignore(c_style_comment)
    .ignore(cpp_style_comment)
    .ignore(Literal("#include") + ... + LineEnd())  # type: ignore [no-untyped-call]
)


def _flatten_result(
    parse_result: ParseResults, *, _keywords: Sequence[str] = ()
) -> Mapping[Sequence[str], Tuple[int, Optional[FoamFile.Value], int]]:
    ret: MutableMapping[Sequence[str], Tuple[int, Optional[FoamFile.Value], int]] = {}
    start = parse_result.locn_start
    assert isinstance(start, int)
    item = parse_result.value
    assert isinstance(item, Sequence)
    end = parse_result.locn_end
    assert isinstance(end, int)
    key, *values = item
    assert isinstance(key, str)
    ret[(*_keywords, key)] = (start, None, end)
    for value in values:
        if isinstance(value, ParseResults):
            ret.update(_flatten_result(value, _keywords=(*_keywords, key)))
        else:
            ret[(*_keywords, key)] = (start, value, end)
    return ret


def _parse(
    contents: str,
) -> Mapping[Sequence[str], Tuple[int, Optional[FoamFile.Value], int]]:
    parse_results = _FILE.parse_string(contents, parse_all=True)
    ret: MutableMapping[Sequence[str], Tuple[int, Optional[FoamFile.Value], int]] = {}
    for parse_result in parse_results:
        ret.update(_flatten_result(parse_result))
    return ret


def _entry_locn(
    parsed: Mapping[Sequence[str], Tuple[int, Optional[FoamFile.Value], int]],
    keywords: Tuple[str, ...],
) -> Tuple[int, int]:
    """
    Location of an entry or where it should be inserted.
    """
    try:
        start, _, end = parsed[keywords]
    except KeyError:
        if len(keywords) > 1:
            _, _, end = parsed[keywords[:-1]]
            end -= 1
        else:
            end = -1

        start = end

    return start, end


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
        return f"({' '.join(_serialize_value(v) for v in value)})"
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
            return f"{value.name} {_serialize_dimensions(value.dimensions)} {_serialize_value(value.value)}"
        else:
            return f"{_serialize_dimensions(value.dimensions)} {_serialize_value(value.value)}"
    else:
        raise TypeError(f"Not a valid dimensioned value: {type(value)}")


def _serialize_value(
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
