from __future__ import annotations

import os
import shlex
import shutil
import sys
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if sys.version_info >= (3, 9):
    from collections.abc import (
        Callable,
        Collection,
        Coroutine,
        Generator,
        Mapping,
        Sequence,
    )
    from collections.abc import Set as AbstractSet
else:
    from typing import (
        AbstractSet,
        Callable,
        Collection,
        Coroutine,
        Generator,
        Mapping,
        Sequence,
    )

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._base import FoamCaseBase
from ._subprocess import DEVNULL, STDOUT

if TYPE_CHECKING:
    from .._files import FoamFieldFile


class FoamCaseRunBase(FoamCaseBase):
    class TimeDirectory(FoamCaseBase.TimeDirectory):
        @abstractmethod
        def cell_centers(
            self,
        ) -> FoamFieldFile | Coroutine[None, None, FoamFieldFile]:
            raise NotImplementedError

        @property
        @abstractmethod
        def _case(self) -> FoamCaseRunBase:
            raise NotImplementedError

        def _cell_centers_calls(self) -> Generator[Any, None, FoamFieldFile]:
            ret = self["C"]

            if ret not in self:
                yield self._case.run(
                    ["postProcess", "-func", "writeCellCentres", "-time", self.name],
                    cpus=0,
                    log=False,
                )

            return ret

    def __delitem__(self, key: int | float | str) -> None:
        shutil.rmtree(self[key].path)

    @staticmethod
    @abstractmethod
    def _run(
        cmd: Sequence[str | os.PathLike[str]] | str,
        *,
        cpus: int,
        **kwargs: Any,
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _rmtree(
        path: os.PathLike[str] | str, *, ignore_errors: bool = False
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _copytree(
        src: os.PathLike[str] | str,
        dest: os.PathLike[str] | str,
        *,
        symlinks: bool = False,
        ignore: Callable[[os.PathLike[str] | str, Collection[str]], Collection[str]]
        | None = None,
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def clean(self, *, check: bool = False) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def copy(self, dst: os.PathLike[str] | str | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def clone(self, dst: os.PathLike[str] | str | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _prepare(
        self, *, check: bool = True, log: bool = True
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        cmd: Sequence[str | os.PathLike[str]] | str | None = None,
        *,
        parallel: bool | None = None,
        cpus: int | None = None,
        check: bool = True,
        log: bool = True,
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def block_mesh(
        self, *, check: bool = True, log: bool = True
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def decompose_par(
        self, *, check: bool = True, log: bool = True
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def reconstruct_par(
        self, *, check: bool = True, log: bool = True
    ) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    @abstractmethod
    def restore_0_dir(self) -> None | Coroutine[None, None, None]:
        raise NotImplementedError

    def __clean_paths(self) -> AbstractSet[Path]:
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

        paths.update(self.path.glob("log.*"))

        return paths

    def __clone_ignore(
        self,
    ) -> Callable[[os.PathLike[str] | str, Collection[str]], Collection[str]]:
        clean_paths = self.__clean_paths()

        def ignore(
            path: os.PathLike[str] | str, names: Collection[str]
        ) -> Collection[str]:
            paths = {Path(path) / name for name in names}
            return {p.name for p in paths.intersection(clean_paths)}

        return ignore

    def __clean_script(self) -> Path | None:
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

        return script

    def __prepare_script(self) -> Path | None:
        """Return the path to the Allrun.pre script, or None if no prepare script is found."""
        script = self.path / "Allrun.pre"

        if not script.is_file():
            return None

        if sys.argv and Path(sys.argv[0]).absolute() == script.absolute():
            return None

        return script

    def __run_script(self, *, parallel: bool | None) -> Path | None:
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
                    msg = "Both (All)run and (All)run-parallel scripts are present. Please specify parallel argument."
                    raise ValueError(msg)
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

    def __env(self, *, shell: bool) -> Mapping[str, str] | None:
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
        return None

    @contextmanager
    def __output(
        self, cmd: Sequence[str | os.PathLike[str]] | str, *, log: bool
    ) -> Generator[tuple[int | IO[bytes], int | IO[bytes]], None, None]:
        if log:
            if isinstance(cmd, str):
                name = shlex.split(cmd)[0]
            else:
                name = Path(cmd[0]).name if isinstance(cmd[0], os.PathLike) else cmd[0]

            with (self.path / f"log.{name}").open("ab") as stdout:
                yield stdout, STDOUT
        else:
            yield DEVNULL, DEVNULL

    def __mkrundir(self) -> Path:
        d = Path(os.environ["FOAM_RUN"], "foamlib")
        d.mkdir(parents=True, exist_ok=True)
        ret = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir=d))
        ret.rmdir()
        return ret

    def _copy_calls(
        self, dst: os.PathLike[str] | str | None
    ) -> Generator[Any, None, Self]:
        if dst is None:
            dst = self.__mkrundir()

        yield self._copytree(self.path, dst, symlinks=True)

        return type(self)(dst)

    def _clean_calls(self, *, check: bool) -> Generator[Any, None, None]:
        script_path = self.__clean_script()

        if script_path is not None:
            yield self.run([script_path], cpus=0, check=check, log=False)
        else:
            for p in self.__clean_paths():
                if p.is_dir():
                    yield self._rmtree(p)
                else:
                    p.unlink()

    def _clone_calls(
        self, dst: os.PathLike[str] | str | None
    ) -> Generator[Any, None, Self]:
        if dst is None:
            dst = self.__mkrundir()

        if self.__clean_script() is not None:
            yield self.copy(dst)
            yield type(self)(dst).clean()
        else:
            yield self._copytree(
                self.path, dst, symlinks=True, ignore=self.__clone_ignore()
            )

        return type(self)(dst)

    def _restore_0_dir_calls(self) -> Generator[Any, None, None]:
        yield self._rmtree(self.path / "0", ignore_errors=True)
        yield self._copytree(self.path / "0.orig", self.path / "0", symlinks=True)

    def _block_mesh_calls(
        self, *, check: bool, log: bool
    ) -> Generator[Any, None, None]:
        yield self.run(["blockMesh"], cpus=0, check=check, log=log)

    def _decompose_par_calls(
        self, *, check: bool, log: bool
    ) -> Generator[Any, None, None]:
        yield self.run(["decomposePar"], cpus=0, check=check, log=log)

    def _reconstruct_par_calls(
        self, *, check: bool, log: bool
    ) -> Generator[Any, None, None]:
        yield self.run(["reconstructPar"], cpus=0, check=check, log=log)

    def _prepare_calls(self, *, check: bool, log: bool) -> Generator[Any, None, None]:
        script_path = self.__prepare_script()

        if script_path is not None:
            yield self.run([script_path], log=log, check=check)

        elif (self.path / "system" / "blockMeshDict").is_file():
            yield self.block_mesh(check=check, log=log)

    def _run_calls(
        self,
        cmd: Sequence[str | os.PathLike[str]] | str | None = None,
        *,
        parallel: bool | None,
        cpus: int | None,
        check: bool,
        log: bool,
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        if cmd is not None:
            if parallel:
                if cpus is None:
                    cpus = max(self._nprocessors, 1)
            else:
                parallel = False
                if cpus is None:
                    cpus = 1

            with self.__output(cmd, log=log) as (stdout, stderr):
                if parallel:
                    if isinstance(cmd, str):
                        cmd = [
                            "mpiexec",
                            "-n",
                            str(cpus),
                            "/bin/sh",
                            "-c",
                            f"{cmd} -parallel",
                        ]
                    else:
                        cmd = ["mpiexec", "-n", str(cpus), *cmd, "-parallel"]

                yield self._run(
                    cmd,
                    cpus=cpus,
                    check=check,
                    cwd=self.path,
                    env=self.__env(shell=isinstance(cmd, str)),
                    stdout=stdout,
                    stderr=stderr,
                    **kwargs,
                )

        else:
            script_path = self.__run_script(parallel=parallel)

            if script_path is not None:
                if parallel or parallel is None:
                    if cpus is None:
                        if self._nprocessors > 0:
                            cpus = self._nprocessors
                        elif (self.path / "system" / "decomposeParDict").is_file():
                            cpus = self._nsubdomains
                        else:
                            cpus = 1
                elif cpus is None:
                    cpus = 1

                yield self.run(
                    [script_path], parallel=False, cpus=cpus, check=check, **kwargs
                )

            else:
                yield self._prepare(check=check, log=log)

                if not self and (self.path / "0.orig").is_dir():
                    yield self.restore_0_dir()

                if parallel is None:
                    parallel = (
                        (cpus is not None and cpus > 1)
                        or self._nprocessors > 0
                        or (self.path / "system" / "decomposeParDict").is_file()
                    )

                if parallel:
                    if (
                        self._nprocessors == 0
                        and (self.path / "system" / "decomposeParDict").is_file()
                    ):
                        yield self.decompose_par(check=check)

                    if cpus is None:
                        cpus = max(self._nprocessors, 1)
                elif cpus is None:
                    cpus = 1

                yield self.run(
                    [self.application],
                    parallel=parallel,
                    cpus=cpus,
                    check=check,
                    **kwargs,
                )
