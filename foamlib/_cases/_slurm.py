from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from ._async import AsyncFoamCase
from ._subprocess import run_async

if TYPE_CHECKING:
    import os


class AsyncSlurmFoamCase(AsyncFoamCase):
    """An asynchronous OpenFOAM case that launches jobs on a Slurm cluster."""

    @staticmethod
    async def _run(
        cmd: Sequence[str | os.PathLike[str]] | str,
        *,
        cpus: int,
        fallback: bool = False,
        **kwargs: Any,
    ) -> None:
        if fallback and shutil.which("salloc") is None:
            await AsyncFoamCase._run(cmd, cpus=cpus, **kwargs)
            return

        if cpus >= 1:
            if isinstance(cmd, str):
                cmd = ["/bin/sh", "-c", cmd]

            if cpus == 1:
                cmd = ["srun", *cmd]

            cmd = ["salloc", "-n", str(cpus), "--job-name", "foamlib", *cmd]

        await run_async(cmd, **kwargs)

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

        :param cmd: The command to run. If None, run the case. If a sequence, the first element is the command and the rest are arguments. If a string, `cmd` is executed in a shell.
        :param parallel: If True, run in parallel using MPI. If None, autodetect whether to run in parallel.
        :param cpus: The number of CPUs to use. If None, autodetect according to the case. If 0, run locally.
        :param check: If True, raise a CalledProcessError if any command returns a non-zero exit code.
        :param log: If True, log the command output to a file.
        :param fallback: If True, fall back to running the command locally if Slurm is not available.
        """
        for coro in self._run_calls(
            cmd=cmd,
            parallel=parallel,
            cpus=cpus,
            check=check,
            log=log,
            fallback=fallback,
        ):
            await coro
