from __future__ import annotations

import asyncio
import multiprocessing
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

if sys.version_info >= (3, 9):
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Collection,
        Iterable,
        Sequence,
    )
else:
    from typing import AsyncGenerator, Awaitable, Collection, Iterable, Sequence

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import aioshutil

from ._base import FoamCaseBase
from ._run import FoamCaseRunBase
from ._subprocess import run_async
from ._util import ValuedGenerator, awaitableasynccontextmanager

if TYPE_CHECKING:
    import os

    from .._files import FoamFieldFile

X = TypeVar("X")
Y = TypeVar("Y")


class AsyncFoamCase(FoamCaseRunBase):
    """
    An OpenFOAM case with asynchronous support.

    Provides methods for running and cleaning cases, as well as accessing files.

    Access the time directories of the case as a sequence, e.g. `case[0]` or `case[-1]`.
    These will return `AsyncFoamCase.TimeDirectory` objects.

    :param path: The path to the case directory. Defaults to the current working
        directory.

    Example usage: ::
        from foamlib import AsyncFoamCase

        case = AsyncFoamCase("path/to/case") # Load an OpenFOAM case
        case[0]["U"].internal_field = [0, 0, 0] # Set the initial velocity field to zero
        await case.run() # Run the case
        for time in case: # Iterate over the time directories
            print(time.time) # Print the time
            print(time["U"].internal_field) # Print the velocity field
    """

    class TimeDirectory(FoamCaseRunBase.TimeDirectory):
        @property
        def _case(self) -> AsyncFoamCase:
            return AsyncFoamCase(self.path.parent)

        async def cell_centers(self) -> FoamFieldFile:
            """
            Write and return the cell centers.

            Currently only works for reconstructed cases (decomposed cases will need to
            be reconstructed first).
            """
            calls = ValuedGenerator(self._cell_centers_calls())

            for coro in calls:
                await coro

            return calls.value

    max_cpus = multiprocessing.cpu_count()
    """
    Maximum number of CPUs to use for running instances of `AsyncFoamCase` concurrently.

    Defaults to the number of CPUs on the system.
    """

    _reserved_cpus = 0
    _cpus_cond = asyncio.Condition()

    @staticmethod
    @asynccontextmanager
    async def _cpus(cpus: int) -> AsyncGenerator[None, None]:
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

    @staticmethod
    async def _run(
        cmd: Sequence[str | os.PathLike[str]] | str,
        *,
        cpus: int,
        **kwargs: Any,
    ) -> None:
        if isinstance(cmd, str):
            cmd = [*AsyncFoamCase._SHELL, cmd]

        async with AsyncFoamCase._cpus(cpus):
            await run_async(cmd, **kwargs)

    @staticmethod
    async def _rmtree(
        path: os.PathLike[str] | str, *, ignore_errors: bool = False
    ) -> None:
        await aioshutil.rmtree(path, ignore_errors=ignore_errors)  # type: ignore [call-arg]

    @staticmethod
    async def _copytree(
        src: os.PathLike[str] | str,
        dest: os.PathLike[str] | str,
        *,
        symlinks: bool = False,
        ignore: Callable[[os.PathLike[str] | str, Collection[str]], Collection[str]]
        | None = None,
    ) -> None:
        await aioshutil.copytree(src, dest, symlinks=symlinks, ignore=ignore)

    async def clean(self, *, check: bool = False) -> None:
        """
        Clean this case.

        If a `clean` or `Allclean` script is present in the case directory, it will be invoked.
        Otherwise, the case directory will be cleaned using these rules:

        - All time directories except `0` will be deleted.
        - The `0` time directory will be deleted if `0.orig` exists.
        - `processor*` directories will be deleted if a `system/decomposeParDict` file is present.
        - `constant/polyMesh` will be deleted if a `system/blockMeshDict` file is present.
        - All `log.*` files will be deleted.

        If this behavior is not appropriate for a case, it is recommended to write a custom
        `clean` script.

        :param check: If True, raise a `CalledProcessError` if the clean script returns a
            non-zero exit code.
        """
        for coro in self._clean_calls(check=check):
            await coro

    @overload
    def __getitem__(self, index: int | float | str) -> AsyncFoamCase.TimeDirectory: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AsyncFoamCase.TimeDirectory]: ...

    def __getitem__(
        self, index: int | slice | float | str
    ) -> AsyncFoamCase.TimeDirectory | Sequence[AsyncFoamCase.TimeDirectory]:
        ret = super().__getitem__(index)
        if isinstance(ret, FoamCaseBase.TimeDirectory):
            return AsyncFoamCase.TimeDirectory(ret)
        return [AsyncFoamCase.TimeDirectory(r) for r in ret]

    async def _prepare(self, *, check: bool = True, log: bool = True) -> None:
        for coro in self._prepare_calls(check=check, log=log):
            await coro

    async def run(
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

        If `cmd` is given, this method will run the given command in the context of the case.

        If `cmd` is `None`, a series of heuristic rules will be used to run the case. This works as
        follows:

        - If a `run`, `Allrun` or `Allrun-parallel` script is present in the case directory,
        it will be invoked. If both `run` and `Allrun` are present, `Allrun` will be used. If
        both `Allrun` and `Allrun-parallel` are present and `parallel` is `None`, an error will
        be raised.
        - If no run script is present but an `Allrun.pre` script exists, it will be invoked.
        - Otherwise, if a `system/blockMeshDict` file is present, the method will call
        `self.block_mesh()`.
        - Then, if a `0.orig` directory is present, it will call `self.restore_0_dir()`.
        - Then, if the case is to be run in parallel (see the `parallel` option) and no
        `processor*` directories exist but a`system/decomposeParDict` file is present, it will
        call `self.decompose_par()`.
        - Then, it will run the case using the application specified in the `controlDict` file.

        If this behavior is not appropriate for a case, it is recommended to write a custom
        `run`, `Allrun`, `Allrun-parallel` or `Allrun.pre` script.

        :param cmd: The command to run. If `None`, run the case. If a sequence, the first element
            is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param parallel: If `True`, run in parallel using MPI. If None, autodetect whether to run
            in parallel.
        :param cpus: The number of CPUs to use. If `None`, autodetect from to the case.
        :param check: If `True`, raise a `CalledProcessError` if any command returns a non-zero
            exit code.
        :param log: If `True`, log the command output to `log.*` files in the case directory.
        """
        for coro in self._run_calls(
            cmd=cmd, parallel=parallel, cpus=cpus, check=check, log=log
        ):
            await coro

    async def block_mesh(self, *, check: bool = True, log: bool = True) -> None:
        """Run blockMesh on this case."""
        for coro in self._block_mesh_calls(check=check, log=log):
            await coro

    async def decompose_par(self, *, check: bool = True, log: bool = True) -> None:
        """Decompose this case for parallel running."""
        for coro in self._decompose_par_calls(check=check, log=log):
            await coro

    async def reconstruct_par(self, *, check: bool = True, log: bool = True) -> None:
        """Reconstruct this case after parallel running."""
        for coro in self._reconstruct_par_calls(check=check, log=log):
            await coro

    async def restore_0_dir(self) -> None:
        """Restore the 0 directory from the 0.orig directory."""
        for coro in self._restore_0_dir_calls():
            await coro

    @awaitableasynccontextmanager
    @asynccontextmanager
    async def copy(
        self, dst: os.PathLike[str] | str | None = None
    ) -> AsyncGenerator[Self, None]:
        """
        Make a copy of this case.

        If used as an asynchronous context manager (i.e., within an `async with` block) the copy
        will be deleted automatically when exiting the block.

        :param dst: The destination path. If `None`, clone to `$FOAM_RUN/foamlib`.

        :return: The copy of the case.

        Example usage: ::
            import os
            from pathlib import Path
            from foamlib import AsyncFoamCase

            pitz_tutorial = AsyncFoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

            my_pitz = await pitz_tutorial.copy("myPitz")
        """
        calls = ValuedGenerator(self._copy_calls(dst))

        for coro in calls:
            await coro

        yield calls.value

        await self._rmtree(calls.value.path)

    @awaitableasynccontextmanager
    @asynccontextmanager
    async def clone(
        self, dst: os.PathLike[str] | str | None = None
    ) -> AsyncGenerator[Self, None]:
        """
        Clone this case (make a clean copy).

        This is equivalent to running `(await self.copy()).clean()`, but it can be more efficient
        in cases that do not contain custom clean scripts.

        If used as an asynchronous context manager (i.e., within an `async with` block) the cloned
        copy will be deleted automatically when exiting the block.

        :param dst: The destination path. If `None`, clone to `$FOAM_RUN/foamlib`.

        :return: The clone of the case.

        Example usage: ::
            import os
            from pathlib import Path
            from foamlib import AsyncFoamCase

            pitz_tutorial = AsyncFoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

            my_pitz = await pitz_tutorial.clone("myPitz")
        """
        calls = ValuedGenerator(self._clone_calls(dst))

        for coro in calls:
            await coro

        yield calls.value

        await self._rmtree(calls.value.path)

    @staticmethod
    def map(coro: Callable[[X], Awaitable[Y]], iterable: Iterable[X]) -> Iterable[Y]:
        """Run an async function on each element of an iterable concurrently."""
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*(coro(arg) for arg in iterable))
        )
