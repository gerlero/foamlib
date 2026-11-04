"""Progress from ``log.*`` files must come from the current run only."""

import sys
from pathlib import Path
from typing import Any

import pytest
from rich.progress import Progress, TaskID

from foamlib import AsyncFoamCase, FoamCase

Updates = list[tuple[float, float | None]]


@pytest.fixture
def updates(monkeypatch: pytest.MonkeyPatch) -> Updates:
    """Record the (completed, total) pairs given to the progress bar."""
    seen: Updates = []
    update = Progress.update

    def spy(self: Progress, task: TaskID, **kwargs: Any) -> None:
        if "completed" in kwargs:
            seen.append((kwargs["completed"], kwargs.get("total")))
        update(self, task, **kwargs)

    monkeypatch.setattr(Progress, "update", spy)
    return seen


@pytest.fixture
def case_path(tmp_path: Path) -> Path:
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "stopAt endTime;\nendTime 10;\n"
    )
    return tmp_path


def run_writing(log: str, content: str) -> list[str]:
    """Command that writes ``content`` to ``log`` in the case directory."""
    code = f"open({log!r}, 'w').write({content!r})"
    return [sys.executable, "-c", code]


def times(updates: Updates) -> list[float]:
    # the last update always marks the run as done
    assert updates[-1] == (1, 1)
    return [completed for completed, _ in updates[:-1]]


def test_progress_ignores_earlier_runs(case_path: Path, updates: Updates) -> None:
    (case_path / "log.old").write_text("Time = 9\n")
    (case_path / "log.append").write_text("Time = 8\n")
    FoamCase(case_path).run(
        [sys.executable, "-c", "print('Time = 1')"], cpus=0, log="log.append"
    )
    seen = times(updates)
    assert 1.0 in seen
    assert 8.0 not in seen
    assert 9.0 not in seen


def test_progress_reads_a_log_rewritten_by_the_run(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "log.script").write_text("Time = 9\n" * 100)
    FoamCase(case_path).run(run_writing("log.script", "Time = 3\n"), cpus=0, log=False)
    seen = times(updates)
    assert 3.0 in seen
    assert 9.0 not in seen


@pytest.mark.asyncio
async def test_async_progress_ignores_earlier_runs(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "log.old").write_text("Time = 9\n")
    await AsyncFoamCase(case_path).run(
        [sys.executable, "-c", "print('Time = 1')"], cpus=0, log=False
    )
    seen = times(updates)
    assert 1.0 in seen
    assert 9.0 not in seen
