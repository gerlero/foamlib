from __future__ import annotations

import asyncio
import contextlib
import os
import selectors
import subprocess
import sys
import time
from io import StringIO, TextIOBase
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Mapping, Sequence
else:
    from typing import Callable, Mapping, Sequence

if sys.version_info >= (3, 10):
    from contextlib import AbstractAsyncContextManager, AbstractContextManager
else:
    from typing import AsyncContextManager as AbstractAsyncContextManager
    from typing import ContextManager as AbstractContextManager

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

CompletedProcess = subprocess.CompletedProcess


class CalledProcessError(subprocess.CalledProcessError):
    @override
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
    stdout: int | TextIOBase = DEVNULL,
    stderr: int | TextIOBase = STDOUT,
    process_stdout: Callable[[str], None] = lambda _: None,
) -> CompletedProcess[str]:
    # Set up log file monitoring
    with LogFileMonitor(case, process_stdout) as log_monitor:
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
                    # Check for new log file content
                    log_monitor.monitor_once()

                    for key, _ in selector.select(
                        timeout=0.1
                    ):  # Small timeout to allow log monitoring
                        assert key.fileobj in open_streams
                        if not (line := key.fileobj.readline()):
                            selector.unregister(key.fileobj)
                            open_streams.remove(key.fileobj)
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

        # Final check for log file content
        log_monitor.monitor_once()

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
    stdout: int | TextIOBase = DEVNULL,
    stderr: int | TextIOBase = STDOUT,
    process_stdout: Callable[[str], None] = lambda _: None,
) -> CompletedProcess[str]:
    # Set up log file monitoring
    async with AsyncLogFileMonitor(case, process_stdout) as log_monitor:
        monitor_task = log_monitor.start_background_monitoring()

        try:
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
                    if not (line := (await proc.stdout.readline()).decode()):
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
                    if not (line := (await proc.stderr.readline()).decode()):
                        break
                    error.write(line)
                    if stderr not in (DEVNULL, PIPE):
                        assert not isinstance(stderr, int)
                        stderr.write(line)

            await asyncio.gather(tee_stdout(), tee_stderr())

            await proc.wait()
            assert proc.returncode is not None

            # Final check for log file content
            await log_monitor.monitor_once_async()

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

        finally:
            # Stop monitoring when done
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task


class LogFileMonitor(AbstractContextManager["LogFileMonitor"]):
    """Monitor log files for progress information."""

    def __init__(
        self,
        case_path: os.PathLike[str],
        process_line: Callable[[str], None] | None = None,
    ) -> None:
        """
        Initialize log file monitor.

        Args:
            case_path: Path to the case directory
            process_line: Callback function to process lines (same signature as in subprocess)
        """
        self.case_path = Path(case_path)
        self.process_line = process_line or (lambda _: None)
        self._monitored_files: dict[Path, int] = {}
        self._monitoring = False

    @override
    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and stop monitoring."""
        self.stop_monitoring()

    def _find_log_files(self) -> list[Path]:
        """Find all log files in the case directory."""
        return list(self.case_path.glob("log.*"))

    def _read_new_lines(self, log_file: Path) -> list[str]:
        """Read new lines from a log file since last check."""
        try:
            current_size = log_file.stat().st_size
            last_size = self._monitored_files.get(log_file, 0)

            if current_size <= last_size:
                return []

            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_size)
                new_content = f.read(current_size - last_size)
                self._monitored_files[log_file] = current_size

                return new_content.splitlines(keepends=True)

        except OSError:
            # File might not exist yet or be temporarily unavailable
            return []

    def monitor_once(self) -> None:
        """Check all log files once for new content and process any Time = lines."""
        log_files = self._find_log_files()

        for log_file in log_files:
            new_lines = self._read_new_lines(log_file)
            for line in new_lines:
                # Process all lines through the callback
                self.process_line(line)

    def start_monitoring(self, interval: float = 0.5) -> None:
        """Start monitoring log files synchronously (for use in sync contexts)."""
        self._monitoring = True
        while self._monitoring:
            self.monitor_once()
            time.sleep(interval)

    def stop_monitoring(self) -> None:
        """Stop monitoring log files."""
        self._monitoring = False


class AsyncLogFileMonitor(
    LogFileMonitor, AbstractAsyncContextManager["AsyncLogFileMonitor"]
):
    """Asynchronous version of log file monitor."""

    def __init__(
        self,
        case_path: os.PathLike[str],
        process_line: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(case_path, process_line)
        self._monitor_task: asyncio.Task[None] | None = None

    @override
    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager and stop monitoring."""
        self.stop_background_monitoring()

    async def monitor_once_async(self) -> None:
        """Async version of monitor_once."""
        # Run the sync version in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.monitor_once)

    async def start_monitoring_async(self, interval: float = 0.5) -> None:
        """Start monitoring log files asynchronously."""
        self._monitoring = True
        try:
            while self._monitoring:
                await self.monitor_once_async()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self._monitoring = False
            raise

    def start_background_monitoring(self, interval: float = 0.5) -> asyncio.Task[None]:
        """Start monitoring in the background as an asyncio task."""
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()

        self._monitor_task = asyncio.create_task(self.start_monitoring_async(interval))
        return self._monitor_task

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._monitoring = False
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
