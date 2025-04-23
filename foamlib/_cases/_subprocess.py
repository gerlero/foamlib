from __future__ import annotations

import asyncio
import os
import selectors
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import IO

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Mapping, Sequence
else:
    from typing import Callable, Mapping, Sequence

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


def _env(case: os.PathLike[str]) -> Mapping[str, str]:
    env = os.environ.copy()

    env["PWD"] = str(Path(case))

    if os.environ.get("FOAM_LD_LIBRARY_PATH", "") and not os.environ.get(
        "DYLD_LIBRARY_PATH", ""
    ):
        env["DYLD_LIBRARY_PATH"] = env["FOAM_LD_LIBRARY_PATH"]

    return env


def run_sync(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    case: os.PathLike[str],
    check: bool = True,
    stdout: int | IO[str] = DEVNULL,
    stderr: int | IO[str] = STDOUT,
    process_stdout: Callable[[str], None] = lambda _: None,
) -> CompletedProcess[str]:
    if sys.version_info < (3, 8):
        cmd = [str(arg) for arg in cmd]

    with subprocess.Popen(
        cmd,
        cwd=case,
        env=_env(case),
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None

        output = StringIO() if stdout is PIPE else None
        error = StringIO()

        if stderr is STDOUT:
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
                        process_stdout(line)
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
    cmd: Sequence[str | os.PathLike[str]],
    *,
    case: os.PathLike[str],
    check: bool = True,
    stdout: int | IO[str] = DEVNULL,
    stderr: int | IO[str] = STDOUT,
    process_stdout: Callable[[str], None] = lambda _: None,
) -> CompletedProcess[str]:
    if sys.version_info < (3, 8):
        cmd = [str(arg) for arg in cmd]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=case,
        env=_env(case),
        stdout=PIPE,
        stderr=PIPE,
    )

    if stderr is STDOUT:
        stderr = stdout

    output = StringIO() if stdout is PIPE else None
    error = StringIO()

    async def tee_stdout() -> None:
        while True:
            assert proc.stdout is not None
            line = (await proc.stdout.readline()).decode()
            if not line:
                break
            process_stdout(line)
            if output is not None:
                output.write(line)
            if stdout not in (DEVNULL, PIPE):
                assert not isinstance(stdout, int)
                stdout.write(line)

    async def tee_stderr() -> None:
        while True:
            assert proc.stderr is not None
            line = (await proc.stderr.readline()).decode()
            if not line:
                break
            error.write(line)
            if stderr not in (DEVNULL, PIPE):
                assert not isinstance(stderr, int)
                stderr.write(line)

    await asyncio.gather(tee_stdout(), tee_stderr())

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
