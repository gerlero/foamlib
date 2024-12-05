from __future__ import annotations

import asyncio
import subprocess
import sys
from io import BytesIO
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    import os

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

CompletedProcess = subprocess.CompletedProcess


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self) -> str:
        if self.stderr:
            if isinstance(self.stderr, bytes):
                return super().__str__() + "\n" + self.stderr.decode()
            if isinstance(self.stderr, str):
                return super().__str__() + "\n" + self.stderr
        return super().__str__()


DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT


def run_sync(
    cmd: Sequence[str | os.PathLike[str]] | str,
    *,
    check: bool = True,
    cwd: os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: int | IO[bytes] | None = None,
    stderr: int | IO[bytes] | None = None,
) -> CompletedProcess[bytes]:
    if not isinstance(cmd, str) and sys.version_info < (3, 8):
        cmd = [str(arg) for arg in cmd]

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=PIPE,
        shell=isinstance(cmd, str),
    )

    if stderr == STDOUT:
        stderr = stdout
    if stderr not in (PIPE, DEVNULL):
        stderr_copy = BytesIO()

        assert not isinstance(stderr, int)
        if stderr is None:
            stderr = sys.stderr.buffer

        assert proc.stderr is not None
        for line in proc.stderr:
            stderr.write(line)
            stderr_copy.write(line)

        output, _ = proc.communicate()
        assert not _
        error = stderr_copy.getvalue()
    else:
        output, error = proc.communicate()

    assert proc.returncode is not None

    if check and proc.returncode != 0:
        raise CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=output,
            stderr=error,
        )

    return CompletedProcess(
        cmd, returncode=proc.returncode, stdout=output, stderr=error
    )


async def run_async(
    cmd: Sequence[str | os.PathLike[str]] | str,
    *,
    check: bool = True,
    cwd: os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: int | IO[bytes] | None = None,
    stderr: int | IO[bytes] | None = None,
) -> CompletedProcess[bytes]:
    if isinstance(cmd, str):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=PIPE,
        )

    else:
        if sys.version_info < (3, 8):
            cmd = [str(arg) for arg in cmd]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=PIPE,
        )

    if stderr == STDOUT:
        stderr = stdout
    if stderr not in (PIPE, DEVNULL):
        stderr_copy = BytesIO()

        assert not isinstance(stderr, int)
        if stderr is None:
            stderr = sys.stderr.buffer

        assert proc.stderr is not None
        async for line in proc.stderr:
            stderr.write(line)
            stderr_copy.write(line)

        output, _ = await proc.communicate()
        assert not _
        error = stderr_copy.getvalue()
    else:
        output, error = await proc.communicate()

    assert proc.returncode is not None

    if check and proc.returncode != 0:
        raise CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=output,
            stderr=error,
        )

    return CompletedProcess(
        cmd, returncode=proc.returncode, stdout=output, stderr=error
    )
