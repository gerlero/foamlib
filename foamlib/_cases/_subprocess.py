import asyncio
import subprocess
import sys
from pathlib import Path
from typing import IO, Any, Optional, Union

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

from .._util import is_sequence

CalledProcessError = subprocess.CalledProcessError

DEVNULL = subprocess.DEVNULL
STDOUT = subprocess.STDOUT


def run_sync(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[Union[int, IO[Any]]] = None,
    stderr: Optional[Union[int, IO[Any]]] = None,
) -> None:
    if sys.version_info < (3, 8):
        if is_sequence(cmd):
            cmd = [str(arg) for arg in cmd]
        else:
            cmd = str(cmd)

    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        shell=not is_sequence(cmd),
        check=check,
    )


async def run_async(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[Union[int, IO[Any]]] = None,
    stderr: Optional[Union[int, IO[Any]]] = None,
) -> None:
    if not is_sequence(cmd):
        proc = await asyncio.create_subprocess_shell(
            str(cmd),
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
