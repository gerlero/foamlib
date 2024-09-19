import os
import shlex
import shutil
import sys
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
        Generator,
        Mapping,
        Sequence,
    )

from ._base import FoamCaseBase
from ._subprocess import DEVNULL, STDOUT


class _FoamCaseRecipes(FoamCaseBase):
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

    def __delitem__(self, key: Union[int, float, str]) -> None:
        shutil.rmtree(self[key].path)

    def _clone_ignore(
        self,
    ) -> Callable[[Union["os.PathLike[str]", str], Collection[str]], Collection[str]]:
        clean_paths = self._clean_paths()

        def ignore(
            path: Union["os.PathLike[str]", str], names: Collection[str]
        ) -> Collection[str]:
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

    @contextmanager
    def _output(
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

    def _copy_cmds(
        self, dest: Union["os.PathLike[str]", str]
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        yield (
            "_copytree",
            (
                self.path,
                dest,
            ),
            {"symlinks": True},
        )

    def _clean_cmds(
        self, *, script: bool = True, check: bool = False
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        script_path = self._clean_script() if script else None

        if script_path is not None:
            yield ("_run", ([script_path],), {"cpus": 0, "check": check, "log": False})
        else:
            for p in self._clean_paths():
                if p.is_dir():
                    yield ("_rmtree", (p,), {})
                else:
                    p.unlink()

    def _clone_cmds(
        self, dest: Union["os.PathLike[str]", str]
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        if self._clean_script() is not None:
            yield ("copy", (dest,), {})
            yield ("clean", (), {})
        else:
            yield (
                "_copytree",
                (
                    self.path,
                    dest,
                ),
                {"symlinks": True, "ignore": self._clone_ignore()},
            )

    def _restore_0_dir_cmds(
        self,
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        yield ("_rmtree", (self.path / "0",), {"ignore_errors": True})
        yield (
            "_copytree",
            (
                self.path / "0.orig",
                self.path / "0",
            ),
            {"symlinks": True},
        )

    def _run_cmds(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
    ) -> Generator[Tuple[str, Sequence[Any], Mapping[str, Any]], None, None]:
        if cmd is not None:
            if parallel:
                if cpus is None:
                    cpus = max(self._nprocessors, 1)
            else:
                parallel = False
                if cpus is None:
                    cpus = 1

            yield ("_run", (cmd,), {"parallel": parallel, "cpus": cpus, "check": check})

        else:
            script_path = self._run_script(parallel=parallel) if script else None

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

                yield (
                    "_run",
                    ([script_path],),
                    {"parallel": False, "cpus": cpus, "check": check},
                )

            else:
                if not self and (self.path / "0.orig").is_dir():
                    yield ("restore_0_dir", (), {})

                if (self.path / "system" / "blockMeshDict").is_file():
                    yield ("block_mesh", (), {"check": check})

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
                        yield ("decompose_par", (), {"check": check})

                    if cpus is None:
                        cpus = max(self._nprocessors, 1)
                else:
                    if cpus is None:
                        cpus = 1

                yield (
                    "_run",
                    ([self.application],),
                    {"parallel": parallel, "cpus": cpus, "check": check},
                )
