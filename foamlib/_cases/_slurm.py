import os
import sys
from typing import Any, Union

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from ._async import AsyncFoamCase
from ._subprocess import run_async


class AsyncSlurmFoamCase(AsyncFoamCase):
    """An asynchronous OpenFOAM case that launches jobs on a Slurm cluster."""

    @staticmethod
    async def _run(
        cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
        *,
        cpus: int,
        **kwargs: Any,
    ) -> None:
        if cpus >= 1:
            if isinstance(cmd, str):
                cmd = ["/bin/sh", "-c", cmd]

            if cpus == 1:
                cmd = ["srun", *cmd]

            cmd = ["salloc", "-n", str(cpus), "--job-name", "foamlib", *cmd]

        await run_async(cmd, **kwargs)
