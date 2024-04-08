import asyncio
import sys
import subprocess

from pathlib import Path
from typing import Any, Union

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

try:
    import numpy as np
except ModuleNotFoundError:
    numpy = False
else:
    numpy = True


def is_sequence(
    value: Any,
) -> TypeGuard[Union["Sequence[Any]", "np.ndarray[Any, Any]"]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        or numpy
        and isinstance(value, np.ndarray)
    )


CalledProcessError = subprocess.CalledProcessError


def run_process(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Union[None, str, Path] = None,
    env: Union[None, Mapping[str, str]] = None,
) -> "subprocess.CompletedProcess[bytes]":
    shell = not is_sequence(cmd)

    if sys.version_info < (3, 8):
        if shell:
            cmd = str(cmd)
        else:
            cmd = (str(arg) for arg in cmd)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        shell=shell,
        check=check,
    )

    return proc


async def run_process_async(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Union[None, str, Path] = None,
    env: Union[None, Mapping[str, str]] = None,
) -> "subprocess.CompletedProcess[bytes]":
    if not is_sequence(cmd):
        proc = await asyncio.create_subprocess_shell(
            str(cmd),
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    else:
        if sys.version_info < (3, 8):
            cmd = (str(arg) for arg in cmd)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    stdout, stderr = await proc.communicate()

    assert proc.returncode is not None

    ret = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    if check:
        ret.check_returncode()

    return ret
