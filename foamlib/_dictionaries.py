from pathlib import Path
from typing import (
    Any,
    Union,
    Sequence,
    Iterator,
    Optional,
    Mapping,
    MutableMapping,
)
from collections import namedtuple
from contextlib import suppress

from ._subprocesses import run_process, CalledProcessError

np: Optional[Any]
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class FoamDictionary(
    MutableMapping[str, Union["FoamDictionary.Value", "FoamDictionary"]]
):
    Value = Union[str, int, float, bool, Sequence["Value"]]
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
    Dimensioned = namedtuple("Dimensioned", ["name", "dimensions", "value"])

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

    @staticmethod
    def _parse_bool(value: str) -> bool:
        if value == "yes":
            return True
        elif value == "no":
            return False
        else:
            raise ValueError(f"Cannot parse '{value}' as a boolean")

    @staticmethod
    def _parse_number(value: str) -> Union[int, float]:
        with suppress(ValueError):
            return int(value)
        with suppress(ValueError):
            return float(value)
        raise ValueError(f"Cannot parse '{value}' as a number")

    @staticmethod
    def _parse_sequence(value: str) -> Sequence[Value]:
        start = value.find("(")
        if start != -1:
            assert value.endswith(")")
            seq = []
            nested = 0
            start += 1
            for i, c in enumerate(value[start:], start=start):
                if c == "(":
                    nested += 1
                elif c == ")":
                    nested -= 1
                if c.isspace() and not nested:
                    v = value[start:i].strip()
                    if v:
                        seq.append(FoamDictionary._parse(v))
                    start = i + 1

            v = value[start:-1].strip()
            if v:
                seq.append(FoamDictionary._parse(v))

            return seq

        else:
            raise ValueError(f"Cannot parse '{value}' as a sequence")

    @staticmethod
    def _parse_field(value: str) -> Value:
        if value.startswith("uniform "):
            value = value[len("uniform ") :]
            return FoamDictionary._parse(value)
        elif value.startswith("nonuniform "):
            value = value[len("nonuniform ") :]
            return FoamDictionary._parse_sequence(value)
        else:
            raise ValueError(f"Cannot parse '{value}' as a field")

    @staticmethod
    def _parse_dimensions(value: str) -> DimensionSet:
        if value.startswith("["):
            assert value.endswith("]")
            return FoamDictionary.DimensionSet(
                *(FoamDictionary._parse_number(v) for v in value[1:-1].split())
            )
        else:
            raise ValueError(f"Cannot parse '{value}' as a dimension set")

    @staticmethod
    def _parse_dimensioned(value: str) -> Dimensioned:
        start = value.find("[", 1)
        if start != -1:
            name = value[:start].strip()
            end = value.find("]", start)
            if end != -1:
                dimensions = FoamDictionary._parse_dimensions(value[start : end + 1])
                value = value[end + 1 :].strip()
                return FoamDictionary.Dimensioned(
                    name, dimensions, FoamDictionary._parse(value)
                )

        raise ValueError(f"Cannot parse '{value}' as a dimensioned value")

    @staticmethod
    def _parse(value: str) -> Value:
        with suppress(ValueError):
            return FoamDictionary._parse_bool(value)

        with suppress(ValueError):
            return FoamDictionary._parse_field(value)

        with suppress(ValueError):
            return FoamDictionary._parse_number(value)

        with suppress(ValueError):
            return FoamDictionary._parse_dimensions(value)

        with suppress(ValueError):
            return FoamDictionary._parse_dimensioned(value)

        with suppress(ValueError):
            return FoamDictionary._parse_sequence(value)

        return value

    @staticmethod
    def _str_mapping(mapping: Any) -> str:
        if isinstance(mapping, FoamDictionary):
            return mapping._cmd(["-value"])
        elif isinstance(mapping, Mapping):
            m = {
                k: FoamDictionary._str(
                    v,
                    assume_field=(k == "internalField" or k == "value"),
                    assume_dimensions=(k == "dimensions"),
                )
                for k, v in mapping.items()
            }
            return f"{{{' '.join(f'{k} {v};' for k, v in m.items())}}}"
        else:
            raise TypeError(f"Not a mapping: {type(mapping)}")

    @staticmethod
    def _str_bool(value: Any) -> str:
        if value is True:
            return "yes"
        elif value is False:
            return "no"
        else:
            raise TypeError(f"Not a bool: {type(value)}")

    @staticmethod
    def _str_sequence(sequence: Any) -> str:
        if (
            isinstance(sequence, Sequence)
            and not isinstance(sequence, str)
            or np
            and isinstance(sequence, np.ndarray)
        ):
            return f"({' '.join(FoamDictionary._str(v) for v in sequence)})"
        else:
            raise TypeError(f"Not a valid sequence: {type(sequence)}")

    @staticmethod
    def _str_field(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"uniform {value}"
        else:
            try:
                s = FoamDictionary._str_sequence(value)
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

    def _str_dimensions(value: Any) -> str:
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and len(value) == 7
        ):
            return f"[{' '.join(str(v) for v in value)}]"
        else:
            raise TypeError(f"Not a valid dimension set: {type(value)}")

    def _str_dimensioned(value: Any) -> str:
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and len(value) == 3
        ):
            return f"{value[0]} {FoamDictionary._str_dimensions(value[1])} {FoamDictionary._str(value[2])}"
        else:
            raise TypeError(f"Not a valid dimensioned value: {type(value)}")

    @staticmethod
    def _str(
        value: Any, assume_field: bool = False, assume_dimensions: bool = False
    ) -> str:
        with suppress(TypeError):
            return FoamDictionary._str_mapping(value)

        if isinstance(value, FoamDictionary.DimensionSet) or assume_dimensions:
            with suppress(TypeError):
                return FoamDictionary._str_dimensions(value)

        if assume_field:
            with suppress(TypeError):
                return FoamDictionary._str_field(value)

        with suppress(TypeError):
            return FoamDictionary._str_dimensioned(value)

        with suppress(TypeError):
            return FoamDictionary._str_sequence(value)

        with suppress(TypeError):
            return FoamDictionary._str_bool(value)

        return str(value)

    def __getitem__(self, key: str) -> Union[Value, "FoamDictionary"]:
        value = self._cmd(["-value"], key=key)

        if value.startswith("{"):
            assert value.endswith("}")
            return FoamDictionary(self._file, [*self._keywords, key])
        else:
            return FoamDictionary._parse(value)

    def __setitem__(self, key: str, value: Any) -> None:
        value = self._str(
            value,
            assume_field=(key == "internalField" or key == "value"),
            assume_dimensions=(key == "dimensions"),
        )

        if len(value) < 1000:
            self._cmd(["-set", value], key=key)
        else:
            self._cmd(["-set", "_foamlib_value_"], key=key)
            contents = self._file.path.read_text()
            contents = contents.replace("_foamlib_value_", value, 1)
            self._file.path.write_text(contents)

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


class FoamFile(FoamDictionary):
    """An OpenFOAM dictionary file as a mutable mapping."""

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(self, [])
        self.path = Path(path).absolute()
        if self.path.is_dir():
            raise IsADirectoryError(self.path)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

    @property
    def dimensions(self) -> FoamDictionary.DimensionSet:
        """
        Alias of `self["dimensions"]`.
        """
        ret = self["dimensions"]
        if not isinstance(ret, FoamDictionary.DimensionSet):
            raise TypeError("dimensions is not a DimensionSet")
        return ret

    @dimensions.setter
    def dimensions(self, value: Any) -> None:
        self["dimensions"] = value

    @property
    def internal_field(self) -> FoamDictionary.Value:
        """
        Alias of `self["internalField"]`.
        """
        ret = self["internalField"]
        if isinstance(ret, FoamDictionary):
            raise TypeError("internalField is a dictionary")
        return ret

    @internal_field.setter
    def internal_field(self, value: Any) -> None:
        self["internalField"] = value

    @property
    def boundary_field(self) -> FoamDictionary:
        """
        Alias of `self["boundaryField"]`.
        """
        ret = self["boundaryField"]
        if not isinstance(ret, FoamDictionary):
            raise TypeError("boundaryField is not a dictionary")
        return ret

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"
