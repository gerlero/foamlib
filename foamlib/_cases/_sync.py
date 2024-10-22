import shutil
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Union, overload

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Sequence
else:
    from typing import Collection, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .._files import FoamFieldFile
from ._base import FoamCaseBase
from ._run import FoamCaseRunBase
from ._subprocess import run_sync
from ._util import ValuedGenerator

if TYPE_CHECKING:
    import os


class FoamCase(FoamCaseRunBase):
    """
    An OpenFOAM case.

    Provides methods for running and cleaning cases, as well as accessing files.

    Access the time directories of the case as a sequence, e.g. `case[0]` or `case[-1]`.

    :param path: The path to the case directory.
    """

    class TimeDirectory(FoamCaseRunBase.TimeDirectory):
        @property
        def _case(self) -> "FoamCase":
            return FoamCase(self.path.parent)

        def cell_centers(self) -> FoamFieldFile:
            """Write and return the cell centers."""
            calls = ValuedGenerator(self._cell_centers_calls())

            for _ in calls:
                pass

            return calls.value

    @staticmethod
    def _run(
        cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
        *,
        cpus: int,
        **kwargs: Any,
    ) -> None:
        run_sync(cmd, **kwargs)

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

    @overload
    def __getitem__(
        self, index: Union[int, float, str]
    ) -> "FoamCase.TimeDirectory": ...

    @overload
    def __getitem__(self, index: slice) -> Sequence["FoamCase.TimeDirectory"]: ...

    def __getitem__(
        self, index: Union[int, slice, float, str]
    ) -> Union["FoamCase.TimeDirectory", Sequence["FoamCase.TimeDirectory"]]:
        ret = super().__getitem__(index)
        if isinstance(ret, FoamCaseBase.TimeDirectory):
            return FoamCase.TimeDirectory(ret)
        return [FoamCase.TimeDirectory(r) for r in ret]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._rmtree(self.path)

    def clean(self, *, check: bool = False) -> None:
        """
        Clean this case.

        :param check: If True, raise a CalledProcessError if the clean script returns a non-zero exit code.
        """
        for _ in self._clean_calls(check=check):
            pass

    def _prepare(self, *, check: bool = True, log: bool = True) -> None:
        for _ in self._prepare_calls(check=check, log=log):
            pass

    def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
        *,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
        log: bool = True,
    ) -> None:
        """
        Run this case, or a specified command in the context of this case.

        :param cmd: The command to run. If None, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param parallel: If True, run in parallel using MPI. If None, autodetect whether to run in parallel.
        :param cpus: The number of CPUs to use. If None, autodetect according to the case.
        :param check: If True, raise a CalledProcessError if any command returns a non-zero exit code.
        :param log: If True, log the command output to a file.
        """
        for _ in self._run_calls(
            cmd=cmd, parallel=parallel, cpus=cpus, check=check, log=log
        ):
            pass

    def block_mesh(self, *, check: bool = True, log: bool = True) -> None:
        """Run blockMesh on this case."""
        for _ in self._block_mesh_calls(check=check, log=log):
            pass

    def decompose_par(self, *, check: bool = True, log: bool = True) -> None:
        """Decompose this case for parallel running."""
        for _ in self._decompose_par_calls(check=check, log=log):
            pass

    def reconstruct_par(self, *, check: bool = True, log: bool = True) -> None:
        """Reconstruct this case after parallel running."""
        for _ in self._reconstruct_par_calls(check=check, log=log):
            pass

    def restore_0_dir(self) -> None:
        """Restore the 0 directory from the 0.orig directory."""
        for _ in self._restore_0_dir_calls():
            pass

    def copy(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> Self:
        """
        Make a copy of this case.

        Use as a context manager to automatically delete the copy when done.

        :param dst: The destination path. If None, clone to `$FOAM_RUN/foamlib`.
        """
        calls = ValuedGenerator(self._copy_calls(dst))

        for _ in calls:
            pass

        return calls.value

    def clone(self, dst: Optional[Union["os.PathLike[str]", str]] = None) -> Self:
        """
        Clone this case (make a clean copy).

        Use as a context manager to automatically delete the clone when done.

        :param dst: The destination path. If None, clone to `$FOAM_RUN/foamlib`.
        """
        calls = ValuedGenerator(self._clone_calls(dst))

        for _ in calls:
            pass

        return calls.value
