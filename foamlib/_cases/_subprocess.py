import asyncio
import os
import subprocess
import sys
from typing import IO, Optional, Union

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

CalledProcessError = subprocess.CalledProcessError
CompletedProcess = subprocess.CompletedProcess

DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT


def run_sync(
    cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
    *,
    check: bool = True,
    cwd: Optional["os.PathLike[str]"] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[Union[int, IO[bytes]]] = None,
    stderr: Optional[Union[int, IO[bytes]]] = None,
) -> "CompletedProcess[bytes]":
    if not isinstance(cmd, str) and sys.version_info < (3, 8):
        cmd = [str(arg) for arg in cmd]

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        shell=isinstance(cmd, str),
        check=check,
    )


async def run_async(
    cmd: Union[Sequence[Union[str, "os.PathLike[str]"]], str],
    *,
    check: bool = True,
    cwd: Optional["os.PathLike[str]"] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[Union[int, IO[bytes]]] = None,
    stderr: Optional[Union[int, IO[bytes]]] = None,
) -> "CompletedProcess[bytes]":
    if isinstance(cmd, str):
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

    else:
        if sys.version_info < (3, 8):
            cmd = [str(arg) for arg in cmd]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

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
