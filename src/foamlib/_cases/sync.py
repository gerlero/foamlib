from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING, Callable, overload

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Sequence
else:
    from typing import Collection, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    import os
    from io import TextIOBase
    from types import TracebackType

from ._run import FoamCaseRunBase
from ._subprocess import DEVNULL, STDOUT, run_sync
from ._util import ValuedGenerator
from .base import FoamCaseBase

if TYPE_CHECKING:
    from .._files import FoamFieldFile


class FoamCase(FoamCaseRunBase):
    """
    An OpenFOAM case with synchronous execution capabilities.

    Extends :class:`FoamCaseBase` to provide methods for running and cleaning cases,
    as well as accessing files.

    Access the time directories of the case as a sequence, e.g. ``case[0]`` or ``case[-1]``.
    These will return :class:`FoamCase.TimeDirectory` objects.

    :param path: The path to the case directory. Defaults to the current working
        directory.

    Example usage: ::

        from foamlib import FoamCase

        case = FoamCase("path/to/case") # Load an OpenFOAM case
        case[0]["U"].internal_field = [0, 0, 0] # Set the initial velocity field to zero
        case.run() # Run the case
        for time in case: # Iterate over the time directories
            print(time.time) # Print the time
            print(time["U"].internal_field) # Print the velocity field
    """

    class TimeDirectory(FoamCaseRunBase.TimeDirectory):
        @override
        @property
        def _case(self) -> FoamCase:
            return FoamCase(self.path.parent)

        @override
        def cell_centers(self) -> FoamFieldFile:
            """
            Write and return the cell centers.

            Currently only works for reconstructed cases (decomposed cases will need to
            be reconstructed first).
            """
            calls = ValuedGenerator(self._cell_centers_calls())

            for _ in calls:
                pass

            return calls.value

    @override
    @staticmethod
    def _run(
        cmd: Sequence[str | os.PathLike[str]] | str,
        *,
        cpus: int,
        case: os.PathLike[str],
        check: bool = True,
        stdout: int | TextIOBase = DEVNULL,
        stderr: int | TextIOBase = STDOUT,
        process_stdout: Callable[[str], None] = lambda _: None,
    ) -> None:
        if isinstance(cmd, str):
            cmd = [*FoamCase._SHELL, cmd]

        run_sync(
            cmd,
            case=case,
            check=check,
            stdout=stdout,
            stderr=stderr,
            process_stdout=process_stdout,
        )

    @override
    @staticmethod
    def _rmtree(path: os.PathLike[str] | str, *, ignore_errors: bool = False) -> None:
        shutil.rmtree(path, ignore_errors=ignore_errors)

    @override
    @staticmethod
    def _copytree(
        src: os.PathLike[str] | str,
        dest: os.PathLike[str] | str,
        *,
        symlinks: bool = False,
        ignore: Callable[[os.PathLike[str] | str, Collection[str]], Collection[str]]
        | None = None,
    ) -> None:
        shutil.copytree(src, dest, symlinks=symlinks, ignore=ignore)

    @overload
    def __getitem__(self, index: int | float | str) -> FoamCase.TimeDirectory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[FoamCase.TimeDirectory]: ...

    @override
    def __getitem__(
        self, index: int | slice | float | str
    ) -> FoamCase.TimeDirectory | Sequence[FoamCase.TimeDirectory]:
        ret = super().__getitem__(index)
        if isinstance(ret, FoamCaseBase.TimeDirectory):
            return FoamCase.TimeDirectory(ret)
        return [FoamCase.TimeDirectory(r) for r in ret]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._rmtree(self.path)

    @override
    def clean(self, *, check: bool = False) -> None:
        """
        Clean this case.

        If a ``clean`` or ``Allclean`` script is present in the case directory, it will be invoked.
        Otherwise, the case directory will be cleaned using these rules:

        - All time directories except ``0`` will be deleted.

        - The ``0`` time directory will be deleted if ``0.orig`` exists.

        - ``processor*`` directories will be deleted if a ``system/decomposeParDict`` file is present.

        - ``constant/polyMesh`` will be deleted if a ``system/blockMeshDict`` file is present.

        - All ``log.*`` files will be deleted.

        If this behavior is not appropriate for a case, it is recommended to write a custom
        ``clean`` script.

        :param check: If True, raise a :class:`CalledProcessError` if the clean script returns a
            non-zero exit code.
        """
        for _ in self._clean_calls(check=check):
            pass

    @override
    def _prepare(self, *, check: bool = True, log: bool = True) -> None:
        for _ in self._prepare_calls(check=check, log=log):
            pass

    @override
    def run(
        self,
        cmd: Sequence[str | os.PathLike[str]] | str | None = None,
        *,
        parallel: bool | None = None,
        cpus: int | None = None,
        check: bool = True,
        log: bool = True,
    ) -> None:
        """
        Run this case, or a specified command in the context of this case.

        If ``cmd`` is given, this method will run the given command in the context of the case.

        If ``cmd`` is ``None``, a series of heuristic rules will be used to run the case. This works as
        follows:

        - If a ``run``, ``Allrun`` or ``Allrun-parallel`` script is present in the case directory,
          it will be invoked. If both ``run`` and ``Allrun`` are present, ``Allrun`` will be used. If
          both ``Allrun`` and ``Allrun-parallel`` are present and :param:`parallel` is ``None``, an error will
          be raised.

        - If no run script is present but an ``Allrun.pre`` script exists, it will be invoked.

        - Otherwise, if a ``system/blockMeshDict`` file is present, the method will call
          :meth:`block_mesh()`.

        - Then, if a ``0.orig`` directory is present, it will call :meth:`restore_0_dir()`.

        - Then, if the case is to be run in parallel (see the :param:`parallel` option) and no
          ``processor*`` directories exist but a ``system/decomposeParDict`` file is present, it will
          call :meth:`decompose_par()`.

        - Then, it will run the case using the application specified in the `controlDict` file.

        If this behavior is not appropriate for a case, it is recommended to write a custom
        ``run``, ``Allrun``, ``Allrun-parallel`` or ``Allrun.pre`` script.

        :param cmd: The command to run. If ``None``, run the case. If a sequence, the first element
            is the command and the rest are arguments. If a string, ``cmd`` is executed in a shell.
        :param parallel: If ``True``, run in parallel using MPI. If None, autodetect whether to run
            in parallel.
        :param cpus: The number of CPUs to use. If ``None``, autodetect from to the case.
        :param check: If ``True``, raise a :class:`CalledProcessError` if any command returns a non-zero
            exit code.
        :param log: If ``True``, log the command output to ``log.*`` files in the case directory.
        """
        for _ in self._run_calls(
            cmd=cmd, parallel=parallel, cpus=cpus, check=check, log=log
        ):
            pass

    @override
    def block_mesh(self, *, check: bool = True, log: bool = True) -> None:
        """Run blockMesh on this case."""
        for _ in self._block_mesh_calls(check=check, log=log):
            pass

    @override
    def decompose_par(self, *, check: bool = True, log: bool = True) -> None:
        """Decompose this case for parallel running."""
        for _ in self._decompose_par_calls(check=check, log=log):
            pass

    @override
    def reconstruct_par(self, *, check: bool = True, log: bool = True) -> None:
        """Reconstruct this case after parallel running."""
        for _ in self._reconstruct_par_calls(check=check, log=log):
            pass

    @override
    def restore_0_dir(self) -> None:
        """Restore the 0 directory from the 0.orig directory."""
        for _ in self._restore_0_dir_calls():
            pass

    @override
    def copy(self, dst: os.PathLike[str] | str | None = None) -> Self:
        """
        Make a copy of this case.

        If used as a context manager (i.e., within a ``with`` block) the copy will be deleted
        automatically when exiting the block.

        :param dst: The destination path. If ``None``, copy to a new directory in ``$FOAM_RUN/foamlib``.

        :return: The copy of the case.

        Example usage: ::

            import os
            from pathlib import Path
            from foamlib import FoamCase

            pitz_tutorial = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

            my_pitz = pitz_tutorial.copy("myPitz")
        """
        calls = ValuedGenerator(self._copy_calls(dst))

        for _ in calls:
            pass

        return calls.value

    @override
    def clone(self, dst: os.PathLike[str] | str | None = None) -> Self:
        """
        Clone this case (make a clean copy).

        This is equivalent to running ``self.copy().clean()``, but it can be more efficient in cases
        that do not contain custom clean scripts.

        If used as a context manager (i.e., within a ``with`` block) the cloned copy will be deleted
        automatically when exiting the block.

        :param dst: The destination path. If ``None``, clone to a new directory in ``$FOAM_RUN/foamlib``.

        :return: The clone of the case.

        Example usage: ::

            import os
            from pathlib import Path
            from foamlib import FoamCase

            pitz_tutorial = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

            my_pitz = pitz_tutorial.clone("myPitz")
        """
        calls = ValuedGenerator(self._clone_calls(dst))

        for _ in calls:
            pass

        return calls.value
