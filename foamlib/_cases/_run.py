import os
import shlex
import shutil
import sys
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import (
    IO,
    Any,
    Optional,
    Tuple,
    Union,
)

if sys.version_info >= (3, 9):
    from collections.abc import (
        Callable,
        Collection,
        Coroutine,
        Generator,
        Mapping,
        Sequence,
        Set,
    )
else:
    from typing import AbstractSet as Set
    from typing import (
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


class FoamCaseRunBase(FoamCaseBase):
    def __delitem__(self, key: Union[int, float, str]) -> None:
        shutil.rmtree(self[key].path)

    @staticmethod
    @abstractmethod
    def _run(
        cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
        *,
        cpus: int,
        **kwargs: Any,
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _rmtree(
        path: Union["os.PathLike[str]", str], *, ignore_errors: bool = False
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _copytree(
        src: Union["os.PathLike[str]", str],
        dest: Union["os.PathLike[str]", str],
        *,
        symlinks: bool = False,
        ignore: Optional[
            Callable[[Union["os.PathLike[str]", str], Collection[str]], Collection[str]]
        ] = None,
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def clean(self, *, check: bool = False) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def copy(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def clone(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
        *,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
        log: bool = True,
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def block_mesh(
        self, *, check: bool = True
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def decompose_par(
        self, *, check: bool = True
    ) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def restore_0_dir(self) -> Union[None, Coroutine[None, None, None]]:
        raise NotImplementedError

    def __clean_paths(self) -> Set[Path]:
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
    ) -> Callable[[Union["os.PathLike[str]", str], Collection[str]], Collection[str]]:
        clean_paths = self.__clean_paths()

        def ignore(
            path: Union["os.PathLike[str]", str], names: Collection[str]
        ) -> Collection[str]:
            paths = {Path(path) / name for name in names}
            return {p.name for p in paths.intersection(clean_paths)}

        return ignore

    def __clean_script(self) -> Optional[Path]:
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

    def __run_script(self, *, parallel: Optional[bool] = None) -> Optional[Path]:
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

    def __env(self, *, shell: bool) -> Optional[Mapping[str, str]]:
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

    @contextmanager
    def __output(
        self, cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str], *, log: bool
    ) -> Generator[Tuple[Union[int, IO[bytes]], Union[int, IO[bytes]]], None, None]:
        if log:
            if isinstance(cmd, str):
                name = shlex.split(cmd)[0]
            else:
                if isinstance(cmd[0], os.PathLike):
                    name = Path(cmd[0]).name
                else:
                    name = cmd[0]

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
        self, dst: Optional[Union["os.PathLike[str]", str]]
    ) -> Generator[Any, None, Self]:
        if dst is None:
            dst = self.__mkrundir()

        yield self._copytree(self.path, dst, symlinks=True)

        return type(self)(dst)

    def _clean_calls(self, *, check: bool = False) -> Generator[Any, None, None]:
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
        self, dst: Optional[Union["os.PathLike[str]", str]]
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

    def _run_calls(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
        *,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
        log: bool = True,
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
                else:
                    if cpus is None:
                        cpus = 1

                yield self.run([script_path], parallel=False, cpus=cpus, check=check)

            else:
                if not self and (self.path / "0.orig").is_dir():
                    yield self.restore_0_dir()

                if (self.path / "system" / "blockMeshDict").is_file():
                    yield self.block_mesh(check=check)

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
                else:
                    if cpus is None:
                        cpus = 1

                yield self.run(
                    [self.application], parallel=parallel, cpus=cpus, check=check
                )
