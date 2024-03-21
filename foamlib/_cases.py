import os
import asyncio
import multiprocessing
import shutil

from pathlib import Path
from contextlib import asynccontextmanager
from typing import (
    Optional,
    Union,
    Collection,
    Mapping,
    Set,
    Sequence,
    AsyncGenerator,
    Callable,
    Iterator,
    overload,
)

import aioshutil

from ._subprocesses import run_process, run_process_async, CalledProcessError
from ._dictionaries import FoamFile, FoamFieldFile


class FoamCaseBase(Sequence["FoamTimeDirectory"]):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path).absolute()
        if not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} is not a directory")

    @property
    def _times(self) -> Sequence["FoamTimeDirectory"]:
        times = []
        for p in self.path.iterdir():
            if p.is_dir():
                try:
                    float(p.name)
                except ValueError:
                    pass
                else:
                    times.append(FoamTimeDirectory(p))

        times.sort(key=lambda t: t.time)

        return times

    @overload
    def __getitem__(self, index: Union[int, float, str]) -> "FoamTimeDirectory": ...

    @overload
    def __getitem__(self, index: slice) -> Sequence["FoamTimeDirectory"]: ...

    def __getitem__(
        self, index: Union[int, slice, float, str]
    ) -> Union["FoamTimeDirectory", Sequence["FoamTimeDirectory"]]:
        if isinstance(index, str):
            return FoamTimeDirectory(self.path / str(index))
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

        paths: Set[Path] = set()

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

        if has_block_mesh_dict and (self.path / "constant" / "polyMesh").exists():
            paths.add(self.path / "constant" / "polyMesh")

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
        """
        Return the path to the (All)clean script, or None if no clean script is found.
        """
        clean = self.path / "clean"
        all_clean = self.path / "Allclean"

        if clean.is_file():
            return clean
        elif all_clean.is_file():
            return all_clean
        else:
            return None

    def _run_script(self, *, parallel: Optional[bool]) -> Optional[Path]:
        """
        Return the path to the (All)run script, or None if no run script is found.
        """
        run = self.path / "run"
        run_parallel = self.path / "run-parallel"
        all_run = self.path / "Allrun"
        all_run_parallel = self.path / "Allrun-parallel"

        if run.is_file() or all_run.is_file():
            if run_parallel.is_file() or all_run_parallel.is_file():
                if parallel:
                    return run_parallel if run_parallel.is_file() else all_run_parallel
                elif parallel is False:
                    return run if run.is_file() else all_run
                else:
                    raise ValueError(
                        "Both (All)run and (All)run-parallel scripts are present. Please specify parallel argument."
                    )
            return run if run.is_file() else all_run
        elif parallel is not False and (
            run_parallel.is_file() or all_run_parallel.is_file()
        ):
            return run_parallel if run_parallel.is_file() else all_run_parallel
        else:
            return None

    def _env(self) -> Mapping[str, str]:
        """
        Return the environment variables for this case.
        """
        env = os.environ.copy()
        env["PWD"] = str(self.path)
        return env

    def _parallel_cmd(
        self, cmd: Union[Sequence[Union[str, Path]], str, Path]
    ) -> Union[Sequence[Union[str, Path]], str]:
        if isinstance(cmd, str) or not isinstance(cmd, Sequence):
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
        """
        The name of the case.
        """
        return self.path.name

    def file(self, path: Union[Path, str]) -> FoamFile:
        """
        Return a FoamFile object for the given path in the case.
        """
        return FoamFile(self.path / path)

    @property
    def _nsubdomains(self) -> Optional[int]:
        """
        Return the number of subdomains as set in the decomposeParDict, or None if no decomposeParDict is found.
        """
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
        """
        Return the number of processor directories in the case.
        """
        return len(list(self.path.glob("processor*")))

    @property
    def application(self) -> str:
        """
        The application name as set in the controlDict.
        """
        application = self.control_dict["application"]
        if not isinstance(application, str):
            raise TypeError(f"application in {self.control_dict} is not a string")
        return application

    @property
    def control_dict(self) -> FoamFile:
        """
        The controlDict file.
        """
        return self.file("system/controlDict")

    @property
    def fv_schemes(self) -> FoamFile:
        """
        The fvSchemes file.
        """
        return self.file("system/fvSchemes")

    @property
    def fv_solution(self) -> FoamFile:
        """
        The fvSolution file.
        """
        return self.file("system/fvSolution")

    @property
    def decompose_par_dict(self) -> FoamFile:
        """
        The decomposeParDict file.
        """
        return self.file("system/decomposeParDict")

    @property
    def block_mesh_dict(self) -> FoamFile:
        """
        The blockMeshDict file.
        """
        return self.file("system/blockMeshDict")

    @property
    def transport_properties(self) -> FoamFile:
        """
        The transportProperties file.
        """
        return self.file("constant/transportProperties")

    @property
    def turbulence_properties(self) -> FoamFile:
        """
        The turbulenceProperties file.
        """
        return self.file("constant/turbulenceProperties")

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def __str__(self) -> str:
        return str(self.path)


class FoamCase(FoamCaseBase):
    """
    An OpenFOAM case.

    Provides methods for running and cleaning cases, as well as accessing files.

    Access the time directories of the case as a sequence, e.g. `case[0]` or `case[-1]`.

    :param path: The path to the case directory.
    """

    def clean(
        self,
        *,
        script: bool = True,
        check: bool = False,
    ) -> None:
        """
        Clean this case.

        :param script: If True, use an (All)clean script if it exists. If False, ignore any clean scripts.
        :param check: If True, raise a RuntimeError if the clean script returns a non-zero exit code.
        """
        script_path = self._clean_script() if script else None

        if script_path is not None:
            self.run([script_path], check=check)
        else:
            for p in self._clean_paths():
                shutil.rmtree(p)

    def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, Path]], str, Path]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        check: bool = True,
    ) -> None:
        """
        Run this case, or a specified command in the context of this case.

        :param cmd: The command to run. If None, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param script: If True and `cmd` is None, use an (All)run(-parallel) script if it exists for running the case. If False or no run script is found, autodetermine the command(s) needed to run the case.
        :param parallel: If True, run in parallel using MPI. If None, autodetect whether to run in parallel.
        :param check: If True, raise a RuntimeError if any command returns a non-zero exit code.
        """
        if cmd is not None:
            if parallel:
                cmd = self._parallel_cmd(cmd)

            try:
                run_process(
                    cmd,
                    check=check,
                    cwd=self.path,
                    env=self._env(),
                )
            except CalledProcessError as e:
                raise RuntimeError(
                    f"{e.cmd} failed with return code {e.returncode}\n{e.stderr.decode()}"
                ) from None

        else:
            script_path = self._run_script(parallel=parallel) if script else None

            if script_path is not None:
                return self.run([script_path], check=check)

            else:
                if (self.path / "system" / "blockMeshDict").is_file():
                    self.block_mesh()

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
                        self.decompose_par()

                self.run(
                    [self.application],
                    parallel=parallel,
                    check=check,
                )

    def block_mesh(self, *, check: bool = True) -> None:
        """
        Run blockMesh on this case.
        """
        self.run(["blockMesh"], check=check)

    def decompose_par(self, *, check: bool = True) -> None:
        """
        Decompose this case for parallel running.
        """
        self.run(["decomposePar"], check=check)

    def reconstruct_par(self, *, check: bool = True) -> None:
        """
        Reconstruct this case after parallel running.
        """
        self.run(["reconstructPar"], check=check)

    def copy(self, dest: Union[Path, str]) -> "FoamCase":
        """
        Make a copy of this case.

        :param dest: The destination path.
        """
        return FoamCase(shutil.copytree(self.path, dest, symlinks=True))

    def clone(self, dest: Union[Path, str]) -> "FoamCase":
        """
        Clone this case (make a clean copy).

        :param dest: The destination path.
        """
        if self._clean_script() is not None:
            copy = self.copy(dest)
            copy.clean()
            return copy

        dest = Path(dest)

        shutil.copytree(self.path, dest, symlinks=True, ignore=self._clone_ignore())

        return FoamCase(dest)


class AsyncFoamCase(FoamCaseBase):
    """
    An OpenFOAM case with asynchronous support.

    Provides methods for running and cleaning cases, as well as accessing files.

    Access the time directories of the case as a sequence, e.g. `case[0]` or `case[-1]`.

    :param path: The path to the case directory.
    """

    max_cpus = multiprocessing.cpu_count()
    """
    Maximum number of CPUs to use for running `AsyncFoamCase`s concurrently. Defaults to the number of CPUs on the system.
    """

    _reserved_cpus = 0
    _cpus_cond = None  # Cannot be initialized here yet

    @staticmethod
    @asynccontextmanager
    async def _cpus(cpus: int) -> AsyncGenerator[None, None]:
        if AsyncFoamCase._cpus_cond is None:
            AsyncFoamCase._cpus_cond = asyncio.Condition()

        cpus = min(cpus, AsyncFoamCase.max_cpus)
        if cpus > 0:
            async with AsyncFoamCase._cpus_cond:
                await AsyncFoamCase._cpus_cond.wait_for(
                    lambda: AsyncFoamCase.max_cpus - AsyncFoamCase._reserved_cpus
                    >= cpus
                )
                AsyncFoamCase._reserved_cpus += cpus
        try:
            yield
        finally:
            if cpus > 0:
                async with AsyncFoamCase._cpus_cond:
                    AsyncFoamCase._reserved_cpus -= cpus
                    AsyncFoamCase._cpus_cond.notify(cpus)

    async def clean(
        self,
        *,
        script: bool = True,
        check: bool = False,
    ) -> None:
        """
        Clean this case.

        :param script: If True, use an (All)clean script if it exists. If False, ignore any clean scripts.
        :param check: If True, raise a RuntimeError if the clean script returns a non-zero exit code.
        """
        script_path = self._clean_script() if script else None

        if script_path is not None:
            await self.run([script_path], check=check)
        else:
            for p in self._clean_paths():
                await aioshutil.rmtree(p)

    async def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, Path]], str, Path]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
    ) -> None:
        """
        Run this case.

        :param cmd: The command to run. If None, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param script: If True and `cmd` is None, use an (All)run(-parallel) script if it exists for running the case. If False or no run script is found, autodetermine the command(s) needed to run the case.
        :param parallel: If True, run in parallel using MPI. If None, autodetect whether to run in parallel.
        :param cpus: The number of CPUs to reserve for the run. The run will wait until the requested number of CPUs is available. If None, autodetect the number of CPUs to reserve.
        :param check: If True, raise a RuntimeError if a command returns a non-zero exit code.
        """
        if cmd is not None:
            if cpus is None:
                if parallel:
                    cpus = min(self._nprocessors, 1)
                else:
                    cpus = 1

            if parallel:
                cmd = self._parallel_cmd(cmd)

            try:
                async with self._cpus(cpus):
                    await run_process_async(
                        cmd,
                        check=check,
                        cwd=self.path,
                        env=self._env(),
                    )
            except CalledProcessError as e:
                raise RuntimeError(
                    f"{e.cmd} failed with return code {e.returncode}\n{e.stderr.decode()}"
                ) from None

        else:
            script_path = self._run_script(parallel=parallel) if script else None

            if script_path is not None:
                if cpus is None:
                    if self._nprocessors > 0:
                        cpus = self._nprocessors
                    else:
                        nsubdomains = self._nsubdomains
                        if nsubdomains is not None:
                            cpus = nsubdomains
                        else:
                            cpus = 1

                await self.run([script_path], check=check, cpus=cpus)

            else:
                if (self.path / "system" / "blockMeshDict").is_file():
                    await self.block_mesh()

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
                        await self.decompose_par()

                    if cpus is None:
                        cpus = min(self._nprocessors, 1)
                else:
                    if cpus is None:
                        cpus = 1

                await self.run(
                    [self.application],
                    parallel=parallel,
                    check=check,
                    cpus=cpus,
                )

    async def block_mesh(self, *, check: bool = True) -> None:
        """
        Run blockMesh on this case.
        """
        await self.run(["blockMesh"], check=check)

    async def decompose_par(self, *, check: bool = True) -> None:
        """
        Decompose this case for parallel running.
        """
        await self.run(["decomposePar"], check=check)

    async def reconstruct_par(self, *, check: bool = True) -> None:
        """
        Reconstruct this case after parallel running.
        """
        await self.run(["reconstructPar"], check=check)

    async def copy(self, dest: Union[Path, str]) -> "AsyncFoamCase":
        """
        Make a copy of this case.

        :param dest: The destination path.
        """
        return AsyncFoamCase(await aioshutil.copytree(self.path, dest, symlinks=True))

    async def clone(self, dest: Union[Path, str]) -> "AsyncFoamCase":
        """
        Clone this case (make a clean copy).

        :param dest: The destination path.
        """
        if self._clean_script() is not None:
            copy = await self.copy(dest)
            await copy.clean()
            return copy

        dest = Path(dest)

        await aioshutil.copytree(
            self.path, dest, symlinks=True, ignore=self._clone_ignore()
        )

        return AsyncFoamCase(dest)


class FoamTimeDirectory(Mapping[str, FoamFieldFile]):
    """
    An OpenFOAM time directory in a case.

    Use as a mapping to access field files in the directory, e.g. `time["U"]`.

    :param path: The path to the time directory.
    """

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path).absolute()
        if not self.path.is_dir():
            raise NotADirectoryError(f"{self.path} is not a directory")

    @property
    def time(self) -> float:
        """
        The time that corresponds to this directory.
        """
        return float(self.path.name)

    @property
    def name(self) -> str:
        """
        The name of this time directory.
        """
        return self.path.name

    def __getitem__(self, key: str) -> FoamFieldFile:
        try:
            return FoamFieldFile(self.path / key)
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def __iter__(self) -> Iterator[str]:
        for p in self.path.iterdir():
            if p.is_file():
                yield p.name

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def __str__(self) -> str:
        return str(self.path)
