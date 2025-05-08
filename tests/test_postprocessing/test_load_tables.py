from __future__ import annotations

import pytest
from foamlib.postprocessing.load_tables import (
    of_cases,
    load_tables,
    list_outputfiles,
    OutputFile,
)


def test_is_of_case() -> None:
    """Test if a directory is an OpenFOAM case."""

    cases = of_cases("tests/test_postprocessing/Cases")
    assert len(cases) == 9


def test_load_tables() -> None:
    """Test if the load_tables function works correctly."""

    file = OutputFile(file_name="force.dat", folder="forces")
    file.add_time(0)

    table = load_tables(output_file=file, dir_name="tests/test_postprocessing/Cases")
    assert table.shape == (3554, 12)


def test_output_files() -> None:
    """Test if the output_files function works correctly."""

    output_files = list_outputfiles("tests/test_postprocessing/Cases")
    # output_files is a dictionary with keys as file names and values as OutputFile objects
    # key: freeSurface--U_freeSurface.raw
    folders = [key.split("--")[0] for key in output_files.keys()]
    assert folders == [
        "freeSurface",
        "freeSurface",
        "probes",
        "probes",
        "probes",
        "forces",
        "forces",
        "sample1",
        "sample2",
    ]

    files = [key.split("--")[1] for key in output_files.keys()]
    assert files == [
        "U_freeSurface.raw",
        "p_freeSurface.raw",
        "T",
        "U",
        "p",
        "force.dat",
        "moment.dat",
        "centreLine_T.xy",
        "centreLine_U.csv",
    ]
