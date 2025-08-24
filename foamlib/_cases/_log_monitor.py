"""Log file monitoring for progress tracking in cases with redirected output."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Pattern

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Sequence
    PathLike = os.PathLike[str]
else:
    from typing import Callable, Sequence
    if hasattr(os, 'PathLike'):
        try:
            PathLike = os.PathLike[str]
        except TypeError:
            # Python 3.7/3.8 - os.PathLike not subscriptable
            PathLike = os.PathLike
    else:
        PathLike = str


class LogFileMonitor:
    """Monitor log files for progress information."""

    # Pattern to match "Time = <number>" lines in OpenFOAM output
    TIME_PATTERN: Pattern[str] = re.compile(r'^Time = (\S+)', re.MULTILINE)

    def __init__(
        self,
        case_path: PathLike,
        process_line: Callable[[str], None] | None = None
    ):
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

            with log_file.open('r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_size)
                new_content = f.read(current_size - last_size)
                self._monitored_files[log_file] = current_size

                return new_content.splitlines(keepends=True)

        except (OSError, IOError):
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


class AsyncLogFileMonitor(LogFileMonitor):
    """Asynchronous version of log file monitor."""

    def __init__(
        self,
        case_path: PathLike,
        process_line: Callable[[str], None] | None = None
    ):
        super().__init__(case_path, process_line)
        self._monitor_task: asyncio.Task[None] | None = None

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

        self._monitor_task = asyncio.create_task(
            self.start_monitoring_async(interval)
        )
        return self._monitor_task

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._monitoring = False
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()


def should_monitor_log_files(cmd: str | Sequence[str | PathLike]) -> bool:
    """
    Determine if a command is likely to redirect output to log files.

    Returns True for commands that are likely to be Allrun scripts or
    other scripts that use runApplication.
    """
    if isinstance(cmd, str):
        # Shell commands are often scripts
        return True

    if not cmd:
        return False

    cmd_name = Path(cmd[0]).name.lower()

    # Shell invocations
    if cmd_name in ('bash', 'sh', 'zsh', 'csh', 'tcsh'):
        return True

    # Common OpenFOAM run scripts
    script_names = {
        'allrun', 'allrun.pre', 'allrun-parallel',
        'run', 'run-parallel'
    }

    return cmd_name in script_names or cmd_name.endswith('.sh')