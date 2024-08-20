import os
import sys
from pathlib import Path
from typing import (
    Any,
    Optional,
    Tuple,
    Union,
    overload,
)

if sys.version_info >= (3, 9):
    from collections.abc import (
        Callable,
        Collection,
        Generator,
        Iterator,
        Mapping,
        Sequence,
        Set,
    )
else:
    from typing import AbstractSet as Set
    from typing import (
        Callable,
        Collection,
        Generator,
        Iterator,
        Mapping,
        Sequence,
    )


from .._files import FoamFieldFile, FoamFile
from .._util import is_sequence


class FoamCaseBase(Sequence["FoamCaseBase.TimeDirectory"]):
    def __init__(self, path: Union[Path, str] = Path()):
        self.path = Path(path).absolute()

    class TimeDirectory(Set[FoamFieldFile]):
        """
        An OpenFOAM time directory in a case.

        Use to access field files in the directory, e.g. `time["U"]`.

        :param path: The path to the time directory.
        """

        def __init__(self, path: Union[Path, str]):
            self.path = Path(path).absolute()

        @property
        def time(self) -> float:
            """The time that corresponds to this directory."""
            return float(self.path.name)

        @property
        def name(self) -> str:
            """The name of this time directory."""
            return self.path.name

        def __getitem__(self, key: str) -> FoamFieldFile:
            if (self.path / key).is_file():
                return FoamFieldFile(self.path / key)
            elif (self.path / f"{key}.gz").is_file():
                return FoamFieldFile(self.path / f"{key}.gz")
            else:
                raise KeyError(key)

        def __contains__(self, obj: object) -> bool:
            if isinstance(obj, FoamFieldFile):
                return obj.path.parent == self.path
            elif isinstance(obj, str):
                return (self.path / obj).is_file() or (
                    self.path / f"{obj}.gz"
                ).is_file()
            else:
                return False

        def __iter__(self) -> Iterator[FoamFieldFile]:
            for p in self.path.iterdir():
                if p.is_file() and (
                    p.suffix != ".gz" or not p.with_suffix("").is_file()
                ):
                    yield FoamFieldFile(p)

        def __len__(self) -> int:
            return len(list(iter(self)))

        def __fspath__(self) -> str:
            return str(self.path)

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}('{self.path}')"

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
        self, index: Union[int, float, str]
    ) -> "FoamCaseBase.TimeDirectory": ...

    @overload
    def __getitem__(self, index: slice) -> Sequence["FoamCaseBase.TimeDirectory"]: ...

    def __getitem__(
        self, index: Union[int, slice, float, str]
    ) -> Union["FoamCaseBase.TimeDirectory", Sequence["FoamCaseBase.TimeDirectory"]]:
        if isinstance(index, str):
            return FoamCaseBase.TimeDirectory(self.path / index)
        elif isinstance(index, float):
            for time in self._times:
                if time.time == index:
                    return time
            raise IndexError(f"Time {index} not found")
        return self._times[index]

    def __len__(self) -> int:
        return len(self._times)

    def _clean_paths(self) -> Set[Path]:
        has_decompose_par_dict = (self.path / "system" / "decomposeParDict").is_file()
        has_block_mesh_dict = (self.path / "system" / "blockMeshDict").is_file()

        paths = set()

        for p in self.path.iterdir():
            if p.is_dir():
                try:
                    t = float(p.name)
                except ValueError:
                    pass
                else:
                    if t != 0:
                        paths.add(p)

                if has_decompose_par_dict and p.name.startswith("processor"):
                    paths.add(p)

        if (self.path / "0.orig").is_dir() and (self.path / "0").is_dir():
            paths.add(self.path / "0")

        if has_block_mesh_dict and (self.path / "constant" / "polyMesh").exists():
            paths.add(self.path / "constant" / "polyMesh")

        if self._run_script() is not None:
            paths.update(self.path.glob("log.*"))

        return paths

    def _clone_ignore(
        self,
    ) -> Callable[[Union[Path, str], Collection[str]], Collection[str]]:
        clean_paths = self._clean_paths()

        def ignore(path: Union[Path, str], names: Collection[str]) -> Collection[str]:
            paths = {Path(path) / name for name in names}
            return {p.name for p in paths.intersection(clean_paths)}

        return ignore

    def _clean_script(self) -> Optional[Path]:
        """Return the path to the (All)clean script, or None if no clean script is found."""
        clean = self.path / "clean"
        all_clean = self.path / "Allclean"

        if clean.is_file():
            script = clean
        elif all_clean.is_file():
            script = all_clean
        else:
            return None

        if sys.argv and Path(sys.argv[0]).absolute() == script.absolute():
            return None

        return script if Path(sys.argv[0]).absolute() != script.absolute() else None

    def _run_script(self, *, parallel: Optional[bool] = None) -> Optional[Path]:
        """Return the path to the (All)run script, or None if no run script is found."""
        run = self.path / "run"
        run_parallel = self.path / "run-parallel"
        all_run = self.path / "Allrun"
        all_run_parallel = self.path / "Allrun-parallel"

        if run.is_file() or all_run.is_file():
            if run_parallel.is_file() or all_run_parallel.is_file():
                if parallel:
                    script = (
                        run_parallel if run_parallel.is_file() else all_run_parallel
                    )
                elif parallel is False:
                    script = run if run.is_file() else all_run
                else:
                    raise ValueError(
                        "Both (All)run and (All)run-parallel scripts are present. Please specify parallel argument."
                    )
            else:
                script = run if run.is_file() else all_run
        elif parallel is not False and (
            run_parallel.is_file() or all_run_parallel.is_file()
        ):
            script = run_parallel if run_parallel.is_file() else all_run_parallel
        else:
            return None

        if sys.argv and Path(sys.argv[0]).absolute() == script.absolute():
            return None

        return script

    def _env(self, *, shell: bool) -> Optional[Mapping[str, str]]:
        sip_workaround = os.environ.get(
            "FOAM_LD_LIBRARY_PATH", ""
        ) and not os.environ.get("DYLD_LIBRARY_PATH", "")

        if not shell or sip_workaround:
            env = os.environ.copy()

            if not shell:
                env["PWD"] = str(self.path)

            if sip_workaround:
                env["DYLD_LIBRARY_PATH"] = env["FOAM_LD_LIBRARY_PATH"]

            return env
        else:
            return None

    def _run_cmds(
        self,
        cmd: Optional[Union[Sequence[Union[str, Path]], str, Path]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        check: bool = True,
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        if cmd is not None:
            if parallel:
                cmd = self._parallel_cmd(cmd)

            yield ("_run", (cmd,), {"check": check})
        else:
            script_path = self._run_script(parallel=parallel) if script else None

            if script_path is not None:
                yield ("_run", ([script_path],), {"check": check})

            else:
                if not self and (self.path / "0.orig").is_dir():
                    yield ("restore_0_dir", (), {})

                if (self.path / "system" / "blockMeshDict").is_file():
                    yield ("block_mesh", (), {"check": check})

                if parallel is None:
                    parallel = (
                        self._nprocessors > 0
                        or (self.path / "system" / "decomposeParDict").is_file()
                    )

                if parallel:
                    if (
                        self._nprocessors == 0
                        and (self.path / "system" / "decomposeParDict").is_file()
                    ):
                        yield ("decompose_par", (), {"check": check})

                yield (
                    "run",
                    ([self.application],),
                    {"parallel": parallel, "check": check},
                )

    def _parallel_cmd(
        self, cmd: Union[Sequence[Union[str, Path]], str, Path]
    ) -> Union[Sequence[Union[str, Path]], str]:
        if not is_sequence(cmd):
            return f"mpiexec -np {self._nprocessors} {cmd} -parallel"
        else:
            return [
                "mpiexec",
                "-np",
                str(self._nprocessors),
                cmd[0],
                "-parallel",
                *cmd[1:],
            ]

    @property
    def name(self) -> str:
        """The name of the case."""
        return self.path.name

    def file(self, path: Union[Path, str]) -> FoamFile:
        """Return a FoamFile object for the given path in the case."""
        return FoamFile(self.path / path)

    @property
    def _nsubdomains(self) -> Optional[int]:
        """Return the number of subdomains as set in the decomposeParDict, or None if no decomposeParDict is found."""
        try:
            nsubdomains = self.decompose_par_dict["numberOfSubdomains"]
            if not isinstance(nsubdomains, int):
                raise TypeError(
                    f"numberOfSubdomains in {self.decompose_par_dict} is not an integer"
                )
            return nsubdomains
        except FileNotFoundError:
            return None

    @property
    def _nprocessors(self) -> int:
        """Return the number of processor directories in the case."""
        return len(list(self.path.glob("processor*")))

    @property
    def application(self) -> str:
        """The application name as set in the controlDict."""
        application = self.control_dict["application"]
        if not isinstance(application, str):
            raise TypeError(f"application in {self.control_dict} is not a string")
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
