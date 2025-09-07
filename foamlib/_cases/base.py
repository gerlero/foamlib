from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, overload

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Sequence
    from collections.abc import Set as AbstractSet
else:
    from typing import AbstractSet, Iterator, Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    import os

from .._files import FoamFieldFile, FoamFile
from ._util import is_path_relative_to


class FoamCaseBase(Sequence["FoamCaseBase.TimeDirectory"]):
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

    class TimeDirectory(AbstractSet[FoamFieldFile]):
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
        def _case(self) -> FoamCaseBase:
            return FoamCaseBase(self.path.parent)

        @property
        def time(self) -> float:
            """The time that corresponds to this directory, as a float."""
            return float(self.path.name)

        @property
        def name(self) -> str:
            """The name of this time directory (the time as a string)."""
            return self.path.name

        def __getitem__(self, key: str, /) -> FoamFieldFile:
            """Return the field file with the given name in this time directory."""
            if (self.path / f"{key}.gz").is_file() and not (self.path / key).is_file():
                return FoamFieldFile(self.path / f"{key}.gz")
            return FoamFieldFile(self.path / key)

        @override
        def __contains__(self, obj: object, /) -> bool:
            """Return ``True`` if the given field file or name exists in this time directory."""
            if isinstance(obj, FoamFieldFile):
                return obj.path.parent == self.path and obj.path.is_file()
            if isinstance(obj, str):
                return (self.path / obj).is_file() or (
                    self.path / f"{obj}.gz"
                ).is_file()
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
            return len(list(iter(self)))

        def __delitem__(self, name: str, /) -> None:
            """Delete the field file with the given name in this time directory."""
            if (self.path / f"{name}.gz").is_file() and not (
                self.path / name
            ).is_file():
                (self.path / f"{name}.gz").unlink()
            else:
                (self.path / name).unlink()

        def __fspath__(self) -> str:
            return str(self.path)

        @override
        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self.path}')"

        @override
        def __str__(self) -> str:
            return str(self.path)

    @property
    def _times(self) -> Sequence[FoamCaseBase.TimeDirectory]:
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
        self, index: int | float | str, /
    ) -> FoamCaseBase.TimeDirectory: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[FoamCaseBase.TimeDirectory]: ...

    @override
    def __getitem__(
        self,
        index: int | slice | float | str,
        /,
    ) -> FoamCaseBase.TimeDirectory | Sequence[FoamCaseBase.TimeDirectory]:
        """Return the time directory at the given index (``int``), indices (``slice``), name (``str``), or time (``float``)."""
        if isinstance(index, str):
            return FoamCaseBase.TimeDirectory(self.path / index)
        if isinstance(index, float):
            for time in self._times:
                if time.time == index:
                    return time
            msg = f"Time {index} not found"
            raise IndexError(msg)
        return self._times[index]

    @override
    def __iter__(self) -> Iterator[FoamCaseBase.TimeDirectory]:
        """Return an iterator over the time directories in the case."""
        return iter(self._times)

    @override
    def __contains__(self, obj: object, /) -> bool:
        """Return ``True`` if the given time directory, name, or time exists in the case."""
        if isinstance(obj, FoamCaseBase.TimeDirectory):
            return obj in self._times
        if isinstance(obj, str):
            return any(time.name == obj for time in self._times)
        if isinstance(obj, float):
            return any(time.time == obj for time in self._times)
        return False

    @override
    def __len__(self) -> int:
        """Return the number of time directories in the case."""
        return len(self._times)

    def __delitem__(self, key: int | float | str, /) -> None:
        """Delete the time directory at the given index (``int``), name (``str``), or time (``float``)."""
        shutil.rmtree(self[key].path)

    @property
    def name(self) -> str:
        """The name of the case."""
        return self.path.name

    def file(self, path: os.PathLike[str] | str) -> FoamFile:
        """Return a :class:`FoamFile` object for the given path in the case."""
        ret = FoamFile(self.path / path)
        if not is_path_relative_to(ret, self):
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
        control_dict = self.control_dict
        try:
            ret = control_dict["application"]
        except KeyError:
            if "solver" in control_dict:
                return "foamRun"
            if "regionSolvers" in control_dict:
                return "foamMultiRun"
            raise
        else:
            if not isinstance(ret, str):
                msg = f"application in {control_dict} is not a string: {ret}"
                raise TypeError(msg)
            return ret

    @property
    def control_dict(self) -> FoamFile:
        """The controlDict file."""
        return self.file("system/controlDict")

    @property
    def fv_schemes(self) -> FoamFile:
        """The fvSchemes file."""
        return self.file("system/fvSchemes")

    @property
    def fv_solution(self) -> FoamFile:
        """The fvSolution file."""
        return self.file("system/fvSolution")

    @property
    def decompose_par_dict(self) -> FoamFile:
        """The decomposeParDict file."""
        return self.file("system/decomposeParDict")

    @property
    def block_mesh_dict(self) -> FoamFile:
        """The blockMeshDict file."""
        return self.file("system/blockMeshDict")

    @property
    def transport_properties(self) -> FoamFile:
        """The transportProperties file."""
        return self.file("constant/transportProperties")

    @property
    def turbulence_properties(self) -> FoamFile:
        """The turbulenceProperties file."""
        return self.file("constant/turbulenceProperties")

    def __fspath__(self) -> str:
        return str(self.path)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"

    @override
    def __str__(self) -> str:
        return str(self.path)
