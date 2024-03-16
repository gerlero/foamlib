import asyncio
import sys

from pathlib import Path
from typing import Union, Sequence, Mapping
import subprocess
from subprocess import CalledProcessError

__all__ = ["run_process", "run_process_async", "CalledProcessError"]


def run_process(
    cmd: Union[Sequence[Union[str, Path]], str, Path],
    *,
    check: bool = True,
    cwd: Union[None, str, Path] = None,
    env: Union[None, Mapping[str, str]] = None,
) -> "subprocess.CompletedProcess[bytes]":
    shell = isinstance(cmd, str) or not isinstance(cmd, Sequence)

    if sys.version_info < (3, 8):
        if shell:
            cmd = str(cmd)
        else:
            cmd = (str(arg) for arg in cmd)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    if isinstance(cmd, str) or not isinstance(cmd, Sequence):
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
