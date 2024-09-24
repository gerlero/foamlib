import asyncio
import multiprocessing
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional, TypeVar, Union

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

from ._recipes import _FoamCaseRecipes
from ._subprocess import run_async
from ._util import awaitableasynccontextmanager

X = TypeVar("X")
Y = TypeVar("Y")


class AsyncFoamCase(_FoamCaseRecipes):
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

    @staticmethod
    async def _rmtree(
        path: Union["os.PathLike[str]", str], ignore_errors: bool = False
    ) -> None:
        await aioshutil.rmtree(path, ignore_errors=ignore_errors)  # type: ignore [call-arg]

    @staticmethod
    async def _copytree(
        src: Union["os.PathLike[str]", str],
        dest: Union["os.PathLike[str]", str],
        *,
        symlinks: bool = False,
        ignore: Optional[
            Callable[[Union["os.PathLike[str]", str], Collection[str]], Collection[str]]
        ] = None,
    ) -> None:
        await aioshutil.copytree(src, dest, symlinks=symlinks, ignore=ignore)

    async def clean(
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
            await getattr(self, name)(*args, **kwargs)

    async def _run(
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

            async with self._cpus(cpus):
                await run_async(
                    cmd,
                    check=check,
                    cwd=self.path,
                    env=self._env(shell=isinstance(cmd, str)),
                    stdout=stdout,
                    stderr=stderr,
                )

    async def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, "os.PathLike[str]"]], str]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
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
            cmd=cmd, script=script, parallel=parallel, cpus=cpus, check=check
        ):
            await getattr(self, name)(*args, **kwargs)

    async def block_mesh(self, *, check: bool = True) -> None:
        """Run blockMesh on this case."""
        await self.run(["blockMesh"], check=check)

    async def decompose_par(self, *, check: bool = True) -> None:
        """Decompose this case for parallel running."""
        await self.run(["decomposePar"], check=check)

    async def reconstruct_par(self, *, check: bool = True) -> None:
        """Reconstruct this case after parallel running."""
        await self.run(["reconstructPar"], check=check)

    async def restore_0_dir(self) -> None:
        """Restore the 0 directory from the 0.orig directory."""
        for name, args, kwargs in self._restore_0_dir_cmds():
            await getattr(self, name)(*args, **kwargs)

    @awaitableasynccontextmanager
    @asynccontextmanager
    async def copy(
        self, dst: Optional[Union["os.PathLike[str]", str]] = None
    ) -> "AsyncGenerator[Self]":
        """
        Make a copy of this case.

        Use as an async context manager to automatically delete the copy when done.

        :param dst: The destination path. If None, copy to a temporary directory.
        """
        if dst is None:
            dst = Path(tempfile.mkdtemp(), self.name)
            tmp = True
        else:
            tmp = False

        for name, args, kwargs in self._copy_cmds(dst):
            await getattr(self, name)(*args, **kwargs)

        yield type(self)(dst)

        if tmp:
            assert isinstance(dst, Path)
            await self._rmtree(dst.parent)
        else:
            await self._rmtree(dst)

    @awaitableasynccontextmanager
    @asynccontextmanager
    async def clone(
        self, dst: Optional[Union["os.PathLike[str]", str]] = None
    ) -> "AsyncGenerator[Self]":
        """
        Clone this case (make a clean copy).

        Use as an async context manager to automatically delete the clone when done.

        :param dst: The destination path. If None, clone to a temporary directory.
        """
        if dst is None:
            dst = Path(tempfile.mkdtemp(), self.name)
            tmp = True
        else:
            tmp = False

        for name, args, kwargs in self._clone_cmds(dst):
            await getattr(self, name)(*args, **kwargs)

        yield type(self)(dst)

        if tmp:
            assert isinstance(dst, Path)
            await self._rmtree(dst.parent)
        else:
            await self._rmtree(dst)

    @staticmethod
    def map(coro: Callable[[X], Awaitable[Y]], iterable: Iterable[X]) -> Iterable[Y]:
        """Run an async function on each element of an iterable concurrently."""
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*(coro(arg) for arg in iterable))
        )
