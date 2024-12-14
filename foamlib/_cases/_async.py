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

    :param path: The path to the case directory.
    """

    class TimeDirectory(FoamCaseRunBase.TimeDirectory):
        @property
        def _case(self) -> AsyncFoamCase:
            return AsyncFoamCase(self.path.parent)

        async def cell_centers(self) -> FoamFieldFile:
            """Write and return the cell centers."""
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

        :param check: If True, raise a CalledProcessError if the clean script returns a non-zero exit code.
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

        :param cmd: The command to run. If None, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param parallel: If True, run in parallel using MPI. If None, autodetect whether to run in parallel.
        :param cpus: The number of CPUs to use. If None, autodetect according to the case.
        :param check: If True, raise a CalledProcessError if any command returns a non-zero exit code.
        :param log: If True, log the command output to a file.
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

        Use as an async context manager to automatically delete the copy when done.

        :param dst: The destination path. If None, clone to `$FOAM_RUN/foamlib`.
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

        Use as an async context manager to automatically delete the clone when done.

        :param dst: The destination path. If None, clone to `$FOAM_RUN/foamlib`.
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
