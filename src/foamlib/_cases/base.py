import contextlib
import shutil
import sys
from collections.abc import Iterator, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import overload

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


import os

from .._files import FoamFieldFile, FoamFile
from ..typing import SubDictLike


class FoamCaseBase(Sequence["FoamCaseBase.TimeDirectory"], os.PathLike[str]):
    """
    Base class for OpenFOAM cases.

    Provides methods for accessing files and time directories in the case, but does not
    provide methods for running the case or any commands. Users are encouraged to use
    :class:`FoamCase` or :class:`AsyncFoamCase` instead of this class.

    Access the time directories of the case as a sequence, e.g. ``case[0]`` or ``case[-1]``.
    These will return :class:`FoamCaseBase.TimeDirectory` objects.

    :param path: The path to the case directory. Defaults to the current working
        directory.
    """

    def __init__(self, path: os.PathLike[str] | str = Path()) -> None:
        self.path = Path(path).absolute()

    class TimeDirectory(AbstractSet[FoamFieldFile], os.PathLike[str]):
        """
        A time directory in an OpenFOAM case.

        Use to access field files in the directory (e.g. ``time["U"]``). These will be
        returned as :class:`FoamFieldFile` objects.

        It also behaves as a set of :class:`FoamFieldFile` objects (e.g. it can be
        iterated over with ``for field in time: ...``).
        """

        def __init__(self, path: os.PathLike[str] | str) -> None:
            self.path = Path(path).absolute()

        @property
        def _case(self) -> "FoamCaseBase":
            return FoamCaseBase(self.path.parent)

        @property
        def time(self) -> float:
            """The time that corresponds to this directory, as a float."""
            return float(self.path.name)

        @property
        def name(self) -> str:
            """The name of this time directory (the time as a string)."""
            return self.path.name

        def __getitem__(self, name: str, /) -> FoamFieldFile:
            """Return the field file with the given name in this time directory."""
            if (self.path / f"{name}.gz").is_file() and not (
                self.path / name
            ).is_file():
                return FoamFieldFile(self.path / f"{name}.gz")
            return FoamFieldFile(self.path / name)

        def __setitem__(self, key: str, data: SubDictLike, /) -> None:
            """Set the contents of the field file with the given name in this time directory."""
            self._case.path.mkdir(exist_ok=True)
            self.path.mkdir(exist_ok=True)
            self[key][:] = data

        @override
        def __contains__(self, obj: object, /) -> bool:
            """Return ``True`` if the given field file or name exists in this time directory."""
            match obj:
                case FoamFieldFile():
                    return obj.path.parent == self.path and obj.path.is_file()
                case str():
                    return (self.path / obj).is_file() or (
                        self.path / f"{obj}.gz"
                    ).is_file()
                case _:
                    return False

        @override
        def __iter__(self) -> Iterator[FoamFieldFile]:
            """Return an iterator over the field files in this time directory."""
            for p in self.path.iterdir():
                if p.is_file() and (
                    p.suffix != ".gz" or not p.with_suffix("").is_file()
                ):
                    yield FoamFieldFile(p)

        @override
        def __len__(self) -> int:
            """Return the number of field files in this time directory."""
            return sum(1 for _ in iter(self))

        def __delitem__(self, name: str, /) -> None:
            """Delete the field file with the given name in this time directory."""
            self[name].path.unlink()

        @override
        def __fspath__(self) -> str:
            return str(self.path)

        @override
        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self.path}')"

        @override
        def __str__(self) -> str:
            return str(self.path)

    @property
    def _times(self) -> Sequence["FoamCaseBase.TimeDirectory"]:
        times = []
        for p in self.path.iterdir():
            if p.is_dir():
                try:
                    float(p.name)
                except ValueError:
                    pass
                else:
                    times.append(FoamCaseBase.TimeDirectory(p))

        times.sort(key=lambda t: t.time)

        return times

    @overload
    def __getitem__(
        self,
        key: int | float | str,
        /,
    ) -> "FoamCaseBase.TimeDirectory": ...

    @overload
    def __getitem__(
        self,
        key: slice,
        /,
    ) -> Sequence["FoamCaseBase.TimeDirectory"]: ...

    @override
    def __getitem__(
        self,
        key: int | slice | float | str,
        /,
    ) -> "FoamCaseBase.TimeDirectory | Sequence[FoamCaseBase.TimeDirectory]":
        """Return the time directory at the given index (``int``), indices (``slice``), name (``str``), or time (``float``)."""
        match key:
            case int() | slice():
                return self._times[key]
            case str():
                return FoamCaseBase.TimeDirectory(self.path / key)
            case float():
                with contextlib.suppress(FileNotFoundError):
                    for time in self._times:
                        if time.time == key:
                            return time
                return FoamCaseBase.TimeDirectory(self.path / f"{key:g}")
            case _:
                msg = f"Invalid type for case lookup: {type(key)} (expected int, slice, str, or float)"
                raise TypeError(msg)

    @override
    def __iter__(self) -> Iterator["FoamCaseBase.TimeDirectory"]:
        """Return an iterator over the time directories in the case."""
        return iter(self._times)

    @override
    def __contains__(self, obj: object, /) -> bool:
        """Return ``True`` if the given time directory, name, or time exists in the case."""
        match obj:
            case FoamCaseBase.TimeDirectory():
                return obj in self._times
            case str():
                return any(time.name == obj for time in self._times)
            case float():
                return any(time.time == obj for time in self._times)
            case _:
                return False

    @override
    def __len__(self) -> int:
        """Return the number of time directories in the case."""
        return len(self._times)

    def __delitem__(self, key: int | slice | float | str, /) -> None:
        """Delete the time directory at the given index (``int``), indices (``slice``), name (``str``), or time (``float``)."""
        match key:
            case slice():
                for time in self._times[key]:
                    shutil.rmtree(time.path)
            case _:
                shutil.rmtree(self[key].path)

    @property
    def name(self) -> str:
        """The name of the case."""
        return self.path.name

    def file(self, path: os.PathLike[str] | str) -> FoamFile:
        """Return a :class:`FoamFile` object for the given path in the case."""
        ret = FoamFile(self.path / path)
        if not ret.path.is_relative_to(self.path):
            msg = f"Path {ret.path} is outside case path {self.path}\nUse FoamFile({path}) to open a file outside the case."
            raise ValueError(msg)
        return ret

    @property
    def _nsubdomains(self) -> int | None:
        """Return the number of subdomains as set in the :attr:`decompose_par_dict`, or ``None`` if no decomposeParDict is found."""
        try:
            nsubdomains = self.decompose_par_dict["numberOfSubdomains"]
            if not isinstance(nsubdomains, int):
                msg = (
                    f"numberOfSubdomains in {self.decompose_par_dict} is not an integer"
                )
                raise TypeError(msg)
        except FileNotFoundError:
            return None
        else:
            return nsubdomains

    @property
    def _nprocessors(self) -> int:
        """Return the number of processor directories in the case."""
        return len(list(self.path.glob("processor*")))

    @property
    def application(self) -> str:
        """The application name."""
        with self.control_dict as control_dict:
            match control_dict:
                case {"application": str() as app}:
                    return app
                case {"application": _}:
                    msg = f"application in {control_dict} is not a string: {control_dict['application']!r}"
                    raise TypeError(msg)
                case {"solver": _}:
                    return "foamRun"
                case {"regionSolvers": _}:
                    return "foamMultiRun"
                case _:
                    msg = "controlDict does not specify application, solver, or regionSolvers"
                    raise KeyError(msg)

    @property
    def control_dict(self) -> FoamFile:
        """The controlDict file."""
        return self.file("system/controlDict")

    @control_dict.setter
    def control_dict(self, data: SubDictLike) -> None:
        """Set the contents of the controlDict file."""
        file = self.control_dict
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def fv_schemes(self) -> FoamFile:
        """The fvSchemes file."""
        return self.file("system/fvSchemes")

    @fv_schemes.setter
    def fv_schemes(self, data: SubDictLike) -> None:
        """Set the contents of the fvSchemes file."""
        file = self.fv_schemes
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def fv_solution(self) -> FoamFile:
        """The fvSolution file."""
        return self.file("system/fvSolution")

    @fv_solution.setter
    def fv_solution(self, data: SubDictLike) -> None:
        """Set the contents of the fvSolution file."""
        file = self.fv_solution
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def decompose_par_dict(self) -> FoamFile:
        """The decomposeParDict file."""
        return self.file("system/decomposeParDict")

    @decompose_par_dict.setter
    def decompose_par_dict(self, data: SubDictLike) -> None:
        """Set the contents of the decomposeParDict file."""
        file = self.decompose_par_dict
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def block_mesh_dict(self) -> FoamFile:
        """The blockMeshDict file."""
        return self.file("system/blockMeshDict")

    @block_mesh_dict.setter
    def block_mesh_dict(self, data: SubDictLike) -> None:
        """Set the contents of the blockMeshDict file."""
        file = self.block_mesh_dict
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def transport_properties(self) -> FoamFile:
        """The transportProperties file."""
        return self.file("constant/transportProperties")

    @transport_properties.setter
    def transport_properties(self, data: SubDictLike) -> None:
        """Set the contents of the transportProperties file."""
        file = self.transport_properties
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @property
    def turbulence_properties(self) -> FoamFile:
        """The turbulenceProperties file."""
        return self.file("constant/turbulenceProperties")

    @turbulence_properties.setter
    def turbulence_properties(self, data: SubDictLike) -> None:
        """Set the contents of the turbulenceProperties file."""
        file = self.turbulence_properties
        self.path.mkdir(exist_ok=True)
        file.path.parent.mkdir(exist_ok=True)
        file[:] = data

    @override
    def __fspath__(self) -> str:
        return str(self.path)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"

    @override
    def __str__(self) -> str:
        return str(self.path)
