import asyncio
import os
import subprocess
import sys
from typing import IO, Optional, Union

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
            elif isinstance(self.stderr, str):
                return super().__str__() + "\n" + self.stderr
        return super().__str__()


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

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=PIPE,
        shell=isinstance(cmd, str),
    )

    error = b""

    if stderr == STDOUT:
        stderr = stdout
    if stderr not in (PIPE, DEVNULL):
        assert not isinstance(stderr, int)
        if stderr is None:
            stderr = sys.stderr.buffer

        assert proc.stderr is not None
        for line in proc.stderr:
            error += line
            stderr.write(line)

    output, _ = proc.communicate()
    assert not _
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

    error = b""

    if stderr == STDOUT:
        stderr = stdout
    if stderr not in (PIPE, DEVNULL):
        assert not isinstance(stderr, int)
        if stderr is None:
            stderr = sys.stderr.buffer

        assert proc.stderr is not None
        async for line in proc.stderr:
            error += line
            stderr.write(line)

    output, _ = await proc.communicate()
    assert not _
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
