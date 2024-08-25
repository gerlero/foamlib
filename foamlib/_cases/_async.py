import asyncio
import multiprocessing
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Optional,
    Union,
)

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator, Sequence
else:
    from typing import AsyncGenerator, Sequence

import aioshutil

from .._util import is_sequence
from ._base import FoamCaseBase
from ._util import check_returncode


class AsyncFoamCase(FoamCaseBase):
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
        script_path = self._clean_script() if script else None

        if script_path is not None:
            await self.run([script_path], check=check)
        else:
            for p in self._clean_paths():
                if p.is_dir():
                    await aioshutil.rmtree(p)  # type: ignore [call-arg]
                else:
                    p.unlink()

    async def _run(
        self,
        cmd: Union[Sequence[Union[str, Path]], str, Path],
        *,
        check: bool = True,
    ) -> None:
        if not is_sequence(cmd):
            proc = await asyncio.create_subprocess_shell(
                str(cmd),
                cwd=self.path,
                env=self._env(shell=True),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE if check else asyncio.subprocess.DEVNULL,
            )

        else:
            if sys.version_info < (3, 8):
                cmd = (str(arg) for arg in cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.path,
                env=self._env(shell=False),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE if check else asyncio.subprocess.DEVNULL,
            )

        stdout, stderr = await proc.communicate()

        assert stdout is None
        assert proc.returncode is not None

        if check:
            check_returncode(proc.returncode, cmd, stderr.decode())

    async def run(
        self,
        cmd: Optional[Union[Sequence[Union[str, Path]], str, Path]] = None,
        *,
        script: bool = True,
        parallel: Optional[bool] = None,
        cpus: Optional[int] = None,
        check: bool = True,
    ) -> None:
        for name, args, kwargs in self._run_cmds(
            cmd=cmd, script=script, parallel=parallel, check=check
        ):
            if cpus is None:
                if name == "run":
                    if kwargs.get("parallel", False):
                        cpus = max(self._nprocessors, 1)
                    else:
                        cpus = 1
                else:
                    cpus = 0

            async with self._cpus(cpus):
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
        await aioshutil.rmtree(self.path / "0", ignore_errors=True)  # type: ignore [call-arg]
        await aioshutil.copytree(self.path / "0.orig", self.path / "0")

    async def copy(self, dest: Union[Path, str]) -> "AsyncFoamCase":
        """
        Make a copy of this case.

        :param dest: The destination path.
        """
        return AsyncFoamCase(await aioshutil.copytree(self.path, dest, symlinks=True))

    async def clone(self, dest: Union[Path, str]) -> "AsyncFoamCase":
        """
        Clone this case (make a clean copy).

        :param dest: The destination path.
        """
        if self._clean_script() is not None:
            copy = await self.copy(dest)
            await copy.clean()
            return copy

        dest = Path(dest)

        await aioshutil.copytree(
            self.path, dest, symlinks=True, ignore=self._clone_ignore()
        )

        return AsyncFoamCase(dest)
