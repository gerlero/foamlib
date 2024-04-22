import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union
from warnings import warn

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


def is_sequence(
    value: Any,
) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, str)


class CalledProcessError(subprocess.CalledProcessError):
    """Exception raised when a process fails and `check=True`."""

    def __str__(self) -> str:
        msg = super().__str__()
        if self.stderr:
            msg += f"\n{self.stderr}"
        return msg


class CalledProcessWarning(Warning):
    """Warning raised when a process prints to stderr and `check=True`."""


def _check(
    retcode: int,
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    stderr: Optional[str],
) -> None:
    if retcode != 0:
        raise CalledProcessError(retcode, cmd, None, stderr)
    elif stderr:
        warn(f"Command {cmd} printed to stderr.\n{stderr}", CalledProcessWarning)


def _env(cwd: Optional[Union[str, Path]] = None) -> Optional[Mapping[str, str]]:
    if cwd is not None:
        env = os.environ.copy()
        env["PWD"] = str(cwd)
        return env
    else:
        return None


def run_process(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Union[None, str, Path] = None,
) -> None:
    shell = not is_sequence(cmd)

    if sys.version_info < (3, 8):
        if shell:
            cmd = str(cmd)
        else:
            cmd = (str(arg) for arg in cmd)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=_env(cwd) if not shell else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if check else subprocess.DEVNULL,
        text=True,
        shell=shell,
    )

    if check:
        _check(proc.returncode, cmd, proc.stderr)


async def run_process_async(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Union[None, str, Path] = None,
) -> None:
    if not is_sequence(cmd):
        proc = await asyncio.create_subprocess_shell(
            str(cmd),
            cwd=cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE if check else asyncio.subprocess.DEVNULL,
        )

    else:
        if sys.version_info < (3, 8):
            cmd = (str(arg) for arg in cmd)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=_env(cwd),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE if check else asyncio.subprocess.DEVNULL,
        )

    stdout, stderr = await proc.communicate()

    assert stdout is None
    assert proc.returncode is not None

    if check:
        _check(proc.returncode, cmd, stderr.decode())
