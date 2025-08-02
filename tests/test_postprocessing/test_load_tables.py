from __future__ import annotations

import pandas as pd
from foamlib.postprocessing.load_tables import (
    datafile,
    functionobject,
    list_function_objects,
    load_tables,
    of_cases,
)
from foamlib.postprocessing.table_reader import read_catch2_benchmark


def max_height_filter(
    table: pd.DataFrame, parameters: list[dict[str, str]]
) -> pd.DataFrame:
    d = {
        "x": [table["x"].max()],
        "y": [table["y"].max()],
        "z": [table["z"].max()],
    }
    for parameter in parameters:
        category = parameter["category"]
        name = parameter["name"]
        d[category] = [name]
    return pd.DataFrame(d)


def test_is_of_case() -> None:
    """Test if a directory is an OpenFOAM case."""

    cases = of_cases("tests/test_postprocessing/Cases")
    assert len(cases) == 3


def test_load_tables_forces() -> None:
    """Test if the load_tables function works correctly."""

    file = functionobject(file_name="force.dat", folder="forces")

    table = load_tables(source=file, dir_name="tests/test_postprocessing/Cases")

    assert table is not None
    assert sorted(table.columns.tolist()[-2:]) == sorted(["grid", "initHeight"])
    assert sorted(table["grid"].unique()) == sorted(["res1", "res2", "res3"])
    assert sorted(table["initHeight"].unique()) == sorted(["height_02", "height_03"])
    assert table.shape == (1095, 12)


def test_load_tables_surface() -> None:
    """Test if the load_tables function works correctly."""

    file = functionobject(file_name="U_freeSurface.raw", folder="freeSurface")

    table = load_tables(
        source=file,
        dir_name="tests/test_postprocessing/Cases",
        filter_table=max_height_filter,
    )

    assert table is not None
    assert sorted(table.columns.tolist()) == sorted(
        ["x", "y", "z", "grid", "initHeight", "timeValue"]
    )
    assert sorted(table["timeValue"].unique()) == ["0.1", "0.2", "0.4"]
    assert sorted(table["grid"].unique()) == sorted(["res1", "res2", "res3"])
    assert sorted(table["initHeight"].unique()) == sorted(["height_02", "height_03"])
    assert table.shape == (9, 6)


def test_output_files() -> None:
    """Test if the output_files function works correctly."""

    output_files = list_function_objects("tests/test_postprocessing/Cases")
    # output_files is a dictionary with keys as file names and values as OutputFile objects

    folders = [key.split("--")[0] for key in output_files]
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
    files = [output_files[key].file_name for key in output_files]
    assert sorted(files) == sorted(
        [
            "U_freeSurface.raw",
            "p_freeSurface.raw",
            "U",
            "p",
            "T",
            "force.dat",
            "moment.dat",
            "centreLine_T.xy",
            "centreLine_U.csv",
        ]
    )

    folders = [str(output_files[key].folder) for key in output_files]

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


def test_load_catch2_benchmarks() -> None:
    """Test if the load_tables function works correctly for catch2 benchmarks."""

    file = datafile(file_name="explicitOperators.xml", folder=".")
    table = load_tables(
        source=file,
        dir_name="tests/test_postprocessing/Cases",
        reader_fn=read_catch2_benchmark,
    )

    assert table is not None
    assert sorted(table.columns.tolist()) == sorted(
        [
            "test_case",
            "benchmark_name",
            "avg_runtime",
            "section1",
            "section2",
            "grid",
            "initHeight",
        ]
    )
    assert sorted(table["grid"].unique()) == sorted(["res1", "res2", "res3"])
    assert sorted(table["initHeight"].unique()) == sorted(["height_02", "height_03"])
    assert table.shape == (105, 7)
