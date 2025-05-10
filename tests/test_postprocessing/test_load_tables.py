from __future__ import annotations

import pytest
import pandas as pd
from foamlib.postprocessing.load_tables import (
    of_cases,
    load_tables,
    list_outputfiles,
    OutputFile,
)


def max_height_filter(table: pd.DataFrame,parameters: dict[str,str]) -> pd.DataFrame:
    d = {
        "x": [table["x"].max()],
        "y": [table["y"].max()],
        "z": [table["z"].max()],
    }
    d.update(parameters)
    return pd.DataFrame(d)


def test_is_of_case() -> None:
    """Test if a directory is an OpenFOAM case."""

    cases = of_cases("tests/test_postprocessing/Cases")
    assert len(cases) == 3


def test_load_tables_forces() -> None:
    """Test if the load_tables function works correctly."""

    file = OutputFile(file_name="force.dat", folder="forces")
    # file.add_time(0)

    table = load_tables(output_file=file, dir_name="tests/test_postprocessing/Cases")

    

    assert table.columns.tolist()[-2:] == ["grid", "initHeight"]
    assert list(table["grid"].unique()) == ["res1", "res2", "res3"]
    assert list(table["initHeight"].unique()) == ["height_02", "height_03"]
    assert table.shape == (1095, 12)


def test_load_tables_surface() -> None:
    """Test if the load_tables function works correctly."""

    file = OutputFile(file_name="U_freeSurface.raw", folder="freeSurface")

    table = load_tables(
        output_file=file, dir_name="tests/test_postprocessing/Cases", filter=max_height_filter
    )

    assert table.columns.tolist() == ["x", "y", "z","grid", "initHeight", "timeValue"]
    assert list(table["grid"].unique()) == ["res1", "res2", "res3"]
    assert list(table["timeValue"].unique()) == [0.1, 0.2, 0.4]
    assert list(table["initHeight"].unique()) == ["height_02", "height_03"]
    assert table.shape == (9, 6)


def test_output_files() -> None:
    """Test if the output_files function works correctly."""

    output_files = list_outputfiles("tests/test_postprocessing/Cases")
    # output_files is a dictionary with keys as file names and values as OutputFile objects
    # key: freeSurface--U_freeSurface.raw
    folders = [key.split("--")[0] for key in output_files.keys()]
    assert sorted(folders) == sorted(
        [
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
    )

    files = [key.split("--")[1] for key in output_files.keys()]
    assert sorted(files) == sorted(
        [
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
    )
