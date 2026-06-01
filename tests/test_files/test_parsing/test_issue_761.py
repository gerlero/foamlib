"""
Test case for issue #761: writing/reading an unquoted file-path string.

OpenFOAM accepts file paths (e.g. ``./../mech.yaml`` or ``/abs/path``) written
to dictionaries unquoted, but foamlib used to reject them because a word/token
was not allowed to start with ``.`` or ``/``. These paths now parse as a single
string and round-trip unchanged.
"""

from pathlib import Path

import pytest
from foamlib import FoamFile


@pytest.mark.parametrize(
    "path",
    [
        "./../mechanisms/H2/CRECK/syngas-creck.yaml",
        "/abs/path/to/file.yaml",
        "mech/H2/x.yaml",
        ".foo",
        "./",
    ],
)
def test_unquoted_path_parses_as_string(path: str) -> None:
    assert FoamFile.loads(f"p {path};".encode()) == {"p": path}


def test_unquoted_path_dumps_round_trip() -> None:
    path = "./../mechanisms/H2/CRECK/syngas-creck.yaml"
    assert FoamFile.loads(FoamFile.dumps(path)) == path


def test_add_path_entry_round_trips(tmp_path: Path) -> None:
    """The original reporter scenario: write path entries, then read them back."""
    data = {
        "path": "./../mechanisms/H2/CRECK/syngas-creck.yaml",
        "absPath": "/abs/x.yaml",
        "note": "plain",
    }

    file_path = tmp_path / "my_file"
    with FoamFile(file_path) as file:
        for k, v in data.items():
            file.add(k, v)

    read_back = FoamFile(file_path)
    for k, v in data.items():
        assert read_back[k] == v


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (".5", 0.5),
        ("-.5", -0.5),
        ("1.5e-3", 0.0015),
        ("5.", 5.0),
    ],
)
def test_floats_still_parse_as_numbers(text: str, expected: float) -> None:
    """Allowing '.'/'/' word starts must not turn genuine floats into strings."""
    assert FoamFile.loads(f"x {text};".encode()) == {"x": expected}
