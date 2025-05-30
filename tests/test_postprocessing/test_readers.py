from __future__ import annotations

import pytest
from foamlib.postprocessing.table_reader import (
    ReaderNotRegisteredError,
    TableReader,
    extract_column_names,
)

# Constants for file paths
force_file = "tests/test_postprocessing/postProcessing/forces/0/force.dat"
free_surface_p_file = (
    "tests/test_postprocessing/postProcessing/freeSurface/0.1/p_freeSurface.raw"
)
free_surface_u_file = (
    "tests/test_postprocessing/postProcessing/freeSurface/0.1/U_freeSurface.raw"
)
probe_p_file = "tests/test_postprocessing/postProcessing/probes/0/p"
probe_u_file = "tests/test_postprocessing/postProcessing/probes/0/U"
probe_t_file = "tests/test_postprocessing/postProcessing/probes/0/T"
sample_file = "tests/test_postprocessing/postProcessing/sample1/0.1/centreLine_T.xy"
csv_file = "tests/test_postprocessing/postProcessing/sample2/0.1/centreLine_U.csv"

# Test data for headers
HEADER_TEST_CASES = [
    (
        force_file,
        [
            "Time",
            "total_x",
            "total_y",
            "total_z",
            "pressure_x",
            "pressure_y",
            "pressure_z",
            "viscous_x",
            "viscous_y",
            "viscous_z",
        ],
    ),
    (free_surface_p_file, ["x", "y", "z", "p"]),
    (free_surface_u_file, ["x", "y", "z", "U_x", "U_y", "U_z"]),
    (probe_p_file, ["Time", "0"]),
    (probe_u_file, ["Time", "0"]),
    (probe_t_file, ["Time", "0"]),
    (sample_file, None),
    (csv_file, None),
]


@pytest.mark.parametrize(("file_path", "expected_headers"), HEADER_TEST_CASES)
def test_read_headers(file_path: str, expected_headers: list[str]) -> None:
    headers = extract_column_names(file_path)
    assert headers == expected_headers


# Test data for table reading

TABLE_TEST_CASES = [
    (
        force_file,
        (109, 10),
        [
            "Time",
            "total_x",
            "total_y",
            "total_z",
            "pressure_x",
            "pressure_y",
            "pressure_z",
            "viscous_x",
            "viscous_y",
            "viscous_z",
        ],
        [
            "time",
            "Fx",
            "Fy",
            "Fz",
            "p_Fx",
            "p_Fy",
            "p_Fz",
            "visk_Fx",
            "visk_Fy",
            "visk_Fz",
        ],
    ),
    (free_surface_p_file, (278, 4), ["x", "y", "z", "p"], ["x", "y", "z", "p"]),
    (
        free_surface_u_file,
        (278, 6),
        ["x", "y", "z", "U_x", "U_y", "U_z"],
        ["x", "y", "z", "u", "v", "w"],
    ),
    (probe_p_file, (109, 2), ["Time", "0"], ["time", "p"]),
    (probe_u_file, (109, 4), [0, 1, 2, 3], ["time", "u", "v", "w"]),
    (probe_t_file, (109, 2), ["Time", "0"], ["time", "T"]),
    (sample_file, (100, 2), [0, 1], ["x", "y"]),
    (
        csv_file,
        (100, 6),
        ["x", "y", "z", "U_0", "U_1", "U_2"],
        ["a", "b", "c", "d", "e", "f"],
    ),
]


@pytest.mark.parametrize(
    ("file_path", "expected_shape", "expected_columns", "column_names"),
    TABLE_TEST_CASES,
)
def test_read_table(
    file_path: str,
    expected_shape: tuple[int, int],
    expected_columns: list[str],
    column_names: list[str],
) -> None:
    """Test that tables are correctly read and columns match expectations."""
    reader = TableReader()
    table = reader.read(file_path)
    assert table.shape == expected_shape
    assert list(table.columns) == expected_columns

    table = reader.read(file_path, column_names=column_names)
    assert table.shape == expected_shape
    assert list(table.columns) == column_names


def test_missing_file() -> None:
    """Test that an error is raised when a file is missing."""
    reader = TableReader()
    with pytest.raises(FileNotFoundError):
        reader.read("non_existent_file.dat")


def test_invalid_extension() -> None:
    """Test that an error is raised when an invalid file extension is used."""
    reader = TableReader()
    with pytest.raises(ReaderNotRegisteredError):
        reader.read("invalid_file.missing_extension")
