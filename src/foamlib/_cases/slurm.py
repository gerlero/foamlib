from __future__ import annotations

import asyncio
import shutil
import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Sequence
else:
    from typing import Callable, Sequence

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    import os
    from io import TextIOBase

from ._subprocess import DEVNULL, STDOUT, run_async
from .async_ import AsyncFoamCase


class AsyncSlurmFoamCase(AsyncFoamCase):
    """
    An asynchronous OpenFOAM case that launches jobs on a Slurm cluster.

    :class:`AsyncSlurmFoamCase` is a subclass of :class:`AsyncFoamCase`. It provides the same interface,
    as the latter, except that it will launch jobs on a Slurm cluster (using ``salloc`` and
    ``srun``) on the user's behalf when running a case or command.

    :param path: The path to the case directory. Defaults to the current working
        directory.
    """

    @override
    @staticmethod
    async def _run(
        cmd: Sequence[str | os.PathLike[str]] | str,
        *,
        cpus: int,
        case: os.PathLike[str],
        check: bool = True,
        stdout: int | TextIOBase = DEVNULL,
        stderr: int | TextIOBase = STDOUT,
        process_stdout: Callable[[str], None] = lambda _: None,
        fallback: bool = False,
    ) -> None:
        if fallback and shutil.which("salloc") is None:
            await AsyncFoamCase._run(
                cmd,
                case=case,
                check=check,
                stdout=stdout,
                stderr=stderr,
                process_stdout=process_stdout,
                cpus=cpus,
            )
            return

        if isinstance(cmd, str):
            cmd = [*AsyncSlurmFoamCase._SHELL, cmd]

        if cpus >= 1:
            if cpus == 1:
                cmd = ["srun", *cmd]

            cmd = ["salloc", "-n", str(cpus), "--job-name", "foamlib", *cmd]

        await run_async(
            cmd,
            case=case,
            check=check,
            stdout=stdout,
            stderr=stderr,
            process_stdout=process_stdout,
        )

    @override
    async def run(
        self,
        cmd: Sequence[str | os.PathLike[str]] | str | None = None,
        *,
        parallel: bool | None = None,
        cpus: int | None = None,
        check: bool = True,
        log: bool = True,
        fallback: bool = False,
    ) -> None:
        """
        Run this case, or a specified command in the context of this case.

        :param cmd: The command to run. If ``None``, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param parallel: If ``True``, run in parallel using MPI. If ``None``, autodetect whether to run in parallel.
        :param cpus: The number of CPUs to use. If ``None``, autodetect according to the case. If ``0``, run locally.
        :param check: If ``True``, raise a :class:`CalledProcessError` if any command returns a non-zero exit code.
        :param log: If ``True``, log the command output to a file.
        :param fallback: If ``True``, fall back to running the command locally if Slurm is not available.
        """
        for coro in self._run_calls(
            cmd=cmd,
            parallel=parallel,
            cpus=cpus,
            check=check,
            log=log,
            fallback=fallback,
        ):
            assert asyncio.iscoroutine(coro)
            await coro
