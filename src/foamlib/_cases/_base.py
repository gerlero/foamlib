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

from .._files import FoamFieldFile, FoamFile

if TYPE_CHECKING:
    import os


class FoamCaseBase(Sequence["FoamCaseBase.TimeDirectory"]):
    def __init__(self, path: os.PathLike[str] | str = Path()) -> None:
        self.path = Path(path).absolute()

    class TimeDirectory(AbstractSet[FoamFieldFile]):
        """
        An OpenFOAM time directory in a case.

        Use to access field files in the directory, e.g. `time["U"]`.

        :param path: The path to the time directory.
        """

        def __init__(self, path: os.PathLike[str] | str) -> None:
            self.path = Path(path).absolute()

        @property
        def _case(self) -> FoamCaseBase:
            return FoamCaseBase(self.path.parent)

        @property
        def time(self) -> float:
            """The time that corresponds to this directory."""
            return float(self.path.name)

        @property
        def name(self) -> str:
            """The name of this time directory."""
            return self.path.name

        def __getitem__(self, key: str) -> FoamFieldFile:
            if (self.path / f"{key}.gz").is_file() and not (self.path / key).is_file():
                return FoamFieldFile(self.path / f"{key}.gz")
            return FoamFieldFile(self.path / key)

        def __contains__(self, obj: object) -> bool:
            if isinstance(obj, FoamFieldFile):
                return obj.path.parent == self.path and obj.path.is_file()
            if isinstance(obj, str):
                return (self.path / obj).is_file() or (
                    self.path / f"{obj}.gz"
                ).is_file()
            return False

        def __iter__(self) -> Iterator[FoamFieldFile]:
            for p in self.path.iterdir():
                if p.is_file() and (
                    p.suffix != ".gz" or not p.with_suffix("").is_file()
                ):
                    yield FoamFieldFile(p)

        def __len__(self) -> int:
            return len(list(iter(self)))

        def __delitem__(self, key: str) -> None:
            if (self.path / f"{key}.gz").is_file() and not (self.path / key).is_file():
                (self.path / f"{key}.gz").unlink()
            else:
                (self.path / key).unlink()

        def __fspath__(self) -> str:
            return str(self.path)

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self.path}')"

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
    def __getitem__(self, index: int | float | str) -> FoamCaseBase.TimeDirectory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[FoamCaseBase.TimeDirectory]: ...

    def __getitem__(
        self, index: int | slice | float | str
    ) -> FoamCaseBase.TimeDirectory | Sequence[FoamCaseBase.TimeDirectory]:
        if isinstance(index, str):
            return FoamCaseBase.TimeDirectory(self.path / index)
        if isinstance(index, float):
            for time in self._times:
                if time.time == index:
                    return time
            msg = f"Time {index} not found"
            raise IndexError(msg)
        return self._times[index]

    def __len__(self) -> int:
        return len(self._times)

    def __delitem__(self, key: int | float | str) -> None:
        shutil.rmtree(self[key].path)

    @property
    def name(self) -> str:
        """The name of the case."""
        return self.path.name

    def file(self, path: os.PathLike[str] | str) -> FoamFile:
        """Return a FoamFile object for the given path in the case."""
        return FoamFile(self.path / path)

    @property
    def _nsubdomains(self) -> int | None:
        """Return the number of subdomains as set in the decomposeParDict, or None if no decomposeParDict is found."""
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
        """The application name as set in the controlDict."""
        application = self.control_dict["application"]
        if not isinstance(application, str):
            msg = f"application in {self.control_dict} is not a string"
            raise TypeError(msg)
        return application

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

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"

    def __str__(self) -> str:
        return str(self.path)
