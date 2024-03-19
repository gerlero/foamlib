from pathlib import Path
from typing import (
    Any,
    Union,
    Sequence,
    Iterator,
    Optional,
    Mapping,
    MutableMapping,
    Tuple,
)
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
    def _parse(value: str) -> Value:
        if value == "yes":
            return True
        elif value == "no":
            return False

        if value.startswith("uniform "):
            value = value[len("uniform ") :]

        with suppress(ValueError):
            return int(value)

        with suppress(ValueError):
            return float(value)

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

        return value

    @staticmethod
    def _str(value: Union[Value, "FoamDictionary"], assume_field: bool = False) -> str:
        if isinstance(value, FoamDictionary):
            return value._cmd(["-value"])
        elif isinstance(value, Mapping):
            out = "{ "
            for k, v in value.items():
                assume_field = k == "internalField" or k == "value"
                out += f"{k} {FoamDictionary._str(v, assume_field=assume_field)}"
                if not isinstance(v, Mapping):
                    out += "; "
            out += "} "
            return out
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            or np
            and isinstance(value, np.ndarray)
        ):
            out = ""
            if assume_field:
                if len(value) < 10:
                    out += "uniform "
                else:
                    out += "nonuniform List<"
                    if not isinstance(value[0], Sequence) and (
                        not np or not isinstance(value[0], np.ndarray)
                    ):
                        out += "scalar"
                    elif len(value[0]) == 3:
                        out += "vector"
                    elif len(value[0]) == 6:
                        out += "symmTensor"
                    elif len(value[0]) == 9:
                        out += "tensor"
                    else:
                        raise ValueError(
                            f"Unsupported sequence length for field: {len(value[0])}"
                        )
                    out += "> "
            out += "( "
            for v in value:
                out += f"{FoamDictionary._str(v)} "
            out += ") "
            return out
        elif isinstance(value, bool):
            return "yes" if value else "no"
        elif assume_field and isinstance(value, (int, float)):
            return f"uniform {value}"
        else:
            return str(value)

    def __getitem__(self, key: str) -> Union[Value, "FoamDictionary"]:
        value = self._cmd(["-value"], key=key)

        if value.startswith("{"):
            assert value.endswith("}")
            return FoamDictionary(self._file, [*self._keywords, key])
        else:
            return FoamDictionary._parse(value)

    def __setitem__(self, key: str, value: Any) -> None:
        assume_field = False
        if key == "internalField":
            assume_field = True
        elif (
            key == "value"
            and len(self._keywords) == 2
            and self._keywords[0] == "boundaryField"
        ):
            assume_field = True

        value = self._str(value, assume_field=assume_field)

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
