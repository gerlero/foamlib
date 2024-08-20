import shutil
import subprocess
import sys
from pathlib import Path
from typing import (
    Optional,
    Union,
)

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from .._util import is_sequence
from ._base import FoamCaseBase
from ._util import check_returncode


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
        :param check: If True, raise a CalledProcessError if the clean script returns a non-zero exit code.
        """
        script_path = self._clean_script() if script else None

        if script_path is not None:
            self.run([script_path], check=check)
        else:
            for p in self._clean_paths():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

    def _run(
        self,
        cmd: Union[Sequence[Union[str, Path]], str, Path],
        *,
        check: bool = True,
    ) -> None:
        shell = not is_sequence(cmd)

        if sys.version_info < (3, 8):
            if shell:
                cmd = str(cmd)
            else:
                cmd = (str(arg) for arg in cmd)

        proc = subprocess.run(
            cmd,
            cwd=self.path,
            env=self._env(shell=shell),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if check else subprocess.DEVNULL,
            text=True,
            shell=shell,
        )

        if check:
            check_returncode(proc.returncode, cmd, proc.stderr)

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
        shutil.rmtree(self.path / "0", ignore_errors=True)
        shutil.copytree(self.path / "0.orig", self.path / "0")

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
