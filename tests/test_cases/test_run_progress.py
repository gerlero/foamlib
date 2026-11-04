"""Progress reporting of ``run()``, without needing OpenFOAM installed."""

import sys
from pathlib import Path
from typing import Any

import pytest
from rich.progress import Progress, TaskID

from foamlib import FoamCase

Updates = list[tuple[float, float | None]]

CONTROL_DICT = """FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
stopAt endTime;
startTime {start};
endTime {end};
"""


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
        CONTROL_DICT.format(start=0, end=10)
    )
    return tmp_path


def echo(line: str) -> list[str]:
    return [sys.executable, "-c", f"print({line!r})"]


@pytest.mark.parametrize("unit", ["", "s", "CAD"])
def test_progress_with_and_without_a_unit(
    case_path: Path, updates: Updates, unit: str
) -> None:
    FoamCase(case_path).run(echo(f"Time = 5{unit}"), cpus=0, log=False)
    assert (5.0, 10) in updates


def test_progress_from_a_negative_start_time(case_path: Path, updates: Updates) -> None:
    (case_path / "system" / "controlDict").write_text(
        CONTROL_DICT.format(start=-180, end=60)
    )
    FoamCase(case_path).run(echo("Time = -60CAD"), cpus=0, log=False)
    assert (120.0, 240) in updates


def test_progress_keeps_a_positive_start_time(
    case_path: Path, updates: Updates
) -> None:
    (case_path / "system" / "controlDict").write_text(
        CONTROL_DICT.format(start=5, end=10)
    )
    FoamCase(case_path).run(echo("Time = 6s"), cpus=0, log=False)
    assert (6.0, 10) in updates
