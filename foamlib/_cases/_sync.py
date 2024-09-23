import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import TracebackType
from typing import (
    Callable,
    Optional,
    Type,
    Union,
)

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Sequence
else:
    from typing import Collection, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._recipes import _FoamCaseRecipes
from ._subprocess import run_sync


class FoamCase(_FoamCaseRecipes):
    """
    An OpenFOAM case.

    Provides methods for running and cleaning cases, as well as accessing files.

    Access the time directories of the case as a sequence, e.g. `case[0]` or `case[-1]`.

    :param path: The path to the case directory.
    """

    def __init__(self, path: Union["os.PathLike[str]", str] = Path()):
        super().__init__(path)
        self._tmp: Optional[bool] = None

    @staticmethod
    def _rmtree(
        path: Union["os.PathLike[str]", str], *, ignore_errors: bool = False
    ) -> None:
        shutil.rmtree(path, ignore_errors=ignore_errors)

    @staticmethod
    def _copytree(
        src: Union["os.PathLike[str]", str],
        dest: Union["os.PathLike[str]", str],
        *,
        symlinks: bool = False,
        ignore: Optional[
            Callable[[Union["os.PathLike[str]", str], Collection[str]], Collection[str]]
        ] = None,
    ) -> None:
        shutil.copytree(src, dest, symlinks=symlinks, ignore=ignore)

    def __enter__(self) -> "FoamCase":
        if self._tmp is None:
            raise RuntimeError(
                "Cannot use a non-copied/cloned case as a context manager"
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._tmp is not None:
            if self._tmp:
                self._rmtree(self.path.parent)
            else:
                self._rmtree(self.path)
        else:
            raise RuntimeError(
                "Cannot use a non-copied/cloned case as a context manager"
            )

    def clean(
        self,
        *,
        script: bool = True,
        check: bool = False,
    ) -> None:
        """
        Clean this case.

        :param script: If True, use an (All)clean script if it exists. If False, ignore any clean scripts.
        :param check: If True, raise a CalledProcessError if the clean script returns a non-zero exit code.
        """
        for name, args, kwargs in self._clean_cmds(script=script, check=check):
            getattr(self, name)(*args, **kwargs)

    def _run(
        self,
        cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
        *,
        parallel: bool = False,
        cpus: int = 1,
        check: bool = True,
        log: bool = True,
    ) -> None:
        with self._output(cmd, log=log) as (stdout, stderr):
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

            run_sync(
                cmd,
                check=check,
                cwd=self.path,
                env=self._env(shell=isinstance(cmd, str)),
                stdout=stdout,
                stderr=stderr,
            )

    def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
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
        :param check: If True, raise a CalledProcessError if any command returns a non-zero exit code.
        """
        for name, args, kwargs in self._run_cmds(
            cmd=cmd, script=script, parallel=parallel, check=check
        ):
            getattr(self, name)(*args, **kwargs)

    def block_mesh(self, *, check: bool = True) -> None:
        """Run blockMesh on this case."""
        self.run(["blockMesh"], check=check)

    def decompose_par(self, *, check: bool = True) -> None:
        """Decompose this case for parallel running."""
        self.run(["decomposePar"], check=check)

    def reconstruct_par(self, *, check: bool = True) -> None:
        """Reconstruct this case after parallel running."""
        self.run(["reconstructPar"], check=check)

    def restore_0_dir(self) -> None:
        """Restore the 0 directory from the 0.orig directory."""
        for name, args, kwargs in self._restore_0_dir_cmds():
            getattr(self, name)(*args, **kwargs)

    def copy(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> "Self":
        """
        Make a copy of this case.

        Use as a context manager to automatically delete the copy when done.

        :param dst: The destination path. If None, copy to a temporary directory.
        """
        if dst is None:
            dst = Path(tempfile.mkdtemp(), self.name)
            tmp = True
        else:
            tmp = False

        for name, args, kwargs in self._copy_cmds(dst):
            getattr(self, name)(*args, **kwargs)

        ret = type(self)(dst)
        ret._tmp = tmp

        return ret

    def clone(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> "Self":
        """
        Clone this case (make a clean copy).

        Use as a context manager to automatically delete the clone when done.

        :param dst: The destination path. If None, clone to a temporary directory.
        """
        if dst is None:
            dst = Path(tempfile.mkdtemp(), self.name)
            tmp = True
        else:
            tmp = False

        for name, args, kwargs in self._clone_cmds(dst):
            getattr(self, name)(*args, **kwargs)

        ret = type(self)(dst)
        ret._tmp = tmp

        return ret
