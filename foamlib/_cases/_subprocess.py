from __future__ import annotations

import asyncio
import selectors
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

    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=PIPE,
        stderr=PIPE,
        shell=isinstance(cmd, str),
    ) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None

        output = BytesIO() if stdout is PIPE else None
        error = BytesIO()

        if stdout is None:
            stdout = sys.stdout.buffer

        if stderr is None:
            stderr = sys.stderr.buffer
        elif stderr is subprocess.STDOUT:
            stderr = stdout

        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            selector.register(proc.stderr, selectors.EVENT_READ)
            open_streams = {proc.stdout, proc.stderr}
            while open_streams:
                for key, _ in selector.select():
                    assert key.fileobj in open_streams
                    line = key.fileobj.readline()  # type: ignore [union-attr]
                    if not line:
                        selector.unregister(key.fileobj)
                        open_streams.remove(key.fileobj)  # type: ignore [arg-type]
                    elif key.fileobj is proc.stdout:
                        if output is not None:
                            output.write(line)
                        if stdout not in (DEVNULL, PIPE):
                            assert not isinstance(stdout, int)
                            stdout.write(line)
                    else:
                        assert key.fileobj is proc.stderr
                        error.write(line)
                        if stderr not in (DEVNULL, PIPE):
                            assert not isinstance(stderr, int)
                            stderr.write(line)

    assert proc.returncode is not None

    if check and proc.returncode != 0:
        raise CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=output.getvalue() if output is not None else None,
            stderr=error.getvalue(),
        )

    return CompletedProcess(
        cmd,
        returncode=proc.returncode,
        stdout=output.getvalue() if output is not None else None,
        stderr=error.getvalue(),
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
            stdout=PIPE,
            stderr=PIPE,
        )

    else:
        if sys.version_info < (3, 8):
            cmd = [str(arg) for arg in cmd]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=PIPE,
            stderr=PIPE,
        )

    if stdout is None:
        stdout = sys.stdout.buffer

    if stderr is None:
        stderr = sys.stderr.buffer
    elif stderr is subprocess.STDOUT:
        stderr = stdout

    output = BytesIO() if stdout is PIPE else None
    error = BytesIO()

    async def process_stdout() -> None:
        while True:
            assert proc.stdout is not None
            line = await proc.stdout.readline()
            if not line:
                break
            if output is not None:
                output.write(line)
            if stdout not in (DEVNULL, PIPE):
                assert not isinstance(stdout, int)
                stdout.write(line)

    async def process_stderr() -> None:
        while True:
            assert proc.stderr is not None
            line = await proc.stderr.readline()
            if not line:
                break
            error.write(line)
            if stderr not in (DEVNULL, PIPE):
                assert not isinstance(stderr, int)
                stderr.write(line)

    await asyncio.gather(process_stdout(), process_stderr())

    await proc.wait()
    assert proc.returncode is not None

    if check and proc.returncode != 0:
        raise CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=output.getvalue() if output is not None else None,
            stderr=error.getvalue(),
        )

    return CompletedProcess(
        cmd,
        returncode=proc.returncode,
        stdout=output.getvalue() if output is not None else None,
        stderr=error.getvalue(),
    )
