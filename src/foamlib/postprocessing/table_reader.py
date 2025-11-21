"""This module provides a utility class for reading tabular data from files with various extensions."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET

try:
    import pandas as pd
    from defusedxml.ElementTree import parse
except ImportError as e:
    msg = "The postprocessing module requires extra dependencies. Install 'foamlib[postprocessing]' to use it."
    raise ImportError(msg) from e


class ReaderNotRegisteredError(Exception):
    """Exception raised when no reader is registered for a given file extension."""


class TableReader:
    """
    TableReader is a utility class for reading tabular data from files with various extensions.

    It uses a registry pattern to associate file extensions with specific reader functions.

    Attributes:
        _registry (Dict[str, Callable[[str], pd.DataFrame]]): A class-level dictionary that maps
            file extensions (as strings) to reader functions. Each reader function takes a file
            path as input and returns a pandas DataFrame.

    Methods:
        register(extension: str) -> Callable[[Callable[[str], pd.DataFrame]], Callable[[str], pd.DataFrame]]:
            A class method decorator used to register a reader function for a specific file extension.
            The extension is case-insensitive.

        read(filepath: Union[str, Path]) -> pd.DataFrame:
            Reads a file and returns its contents as a pandas DataFrame. The file extension is used
            to determine the appropriate reader function. Raises a ValueError if no reader is registered
            for the file's extension.
    """

    _registry: ClassVar[
        dict[str, Callable[[str | Path, list[str] | None], pd.DataFrame]]
    ] = {}

    def __init__(self) -> None:
        """Initialize the TableReader instance."""

    @classmethod
    def register(
        cls, extension: str
    ) -> Callable[
        [Callable[[str | Path, list[str] | None], pd.DataFrame]],
        Callable[[str | Path, list[str] | None], pd.DataFrame],
    ]:
        """
        Register a reader function for a specific file extension.

        The extension is case-insensitive.

        Args:
            extension (str): The file extension (e.g., ".dat", ".raw") to register the reader for.

        Returns:
            Callable[[Callable[[str | Path, list[str] | None], pd.DataFrame]], Callable[[str | Path, list[str] | None], pd.DataFrame]]:
                A decorator that registers the function as a reader for the specified extension.
        """

        def decorator(
            func: Callable[[str | Path, list[str] | None], pd.DataFrame],
        ) -> Callable[[str | Path, list[str] | None], pd.DataFrame]:
            cls._registry[extension.lower()] = func
            return func

        return decorator

    def read(
        self, filepath: str | Path, column_names: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Read a file and return its contents as a pandas DataFrame.

        The file extension is used to determine the appropriate reader function.

        Raises:
            ValueError: If no reader is registered for the file's extension.

        Args:
            filepath (Union[str, Path]): The path to the file to be read.

        Returns:
            pd.DataFrame: The contents of the file as a pandas DataFrame.
        """
        if (ext := str(Path(filepath).suffix.lower())) not in self._registry:
            error_message = f"No reader registered for extension: '{ext}'"
            raise ReaderNotRegisteredError(error_message)
        return self._registry[ext](filepath, column_names)


def is_convertible_to_float(values: list[str]) -> bool:
    """
    Check if all values in a list are convertible to floats.

    Args:
        values (list[str]): A list of string values to check.
    Returns:
        bool: True if all values are convertible to floats, False otherwise.
    """
    try:
        [float(value) for value in values]
    except ValueError:
        return False
    else:
        return True


def extract_column_names(filepath: str | Path) -> list[str] | None:
    """
    Extract column names from the first 20 lines of a file.

    Args:
        filepath (str): The path to the file from which to extract column names.
    Returns:
        list[str] | None: A list of column names extracted from the file, or None if no
            comment lines are found.
    """
    with Path(filepath).open() as f:
        first_lines = [line.strip() for line in islice(f, 20)]

    # Filter only comment lines
    if not (comment_lines := [line for line in first_lines if line.startswith("#")]):
        return None

    # Take the last comment line and split into column names
    last_comment = comment_lines[-1]
    headers = last_comment.lstrip("#").strip()
    return headers.split()


def update_column_names(
    table: pd.DataFrame, column_names: list[str] | None
) -> pd.DataFrame:
    """
    Update the column names of a DataFrame if provided.

    Args:
        table (pd.DataFrame): The DataFrame to update.
        column_names (Optional[list[str]]): The new column names to set.
    Returns:
        pd.DataFrame: The updated DataFrame with new column names.
    """
    if column_names is not None:
        if len(column_names) != len(table.columns):
            error_message = (
                f"Number of column names ({len(column_names)}) does not match "
                f"number of columns in DataFrame ({len(table.columns)})."
            )
            raise ValueError(error_message)
        table.columns = pd.Index(column_names)
    return table


def read_oftable(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Use a regular expression to parse the file and separate on parentheses and whitespace.

    Args:
        filepath (str): The path to the .oftable file to be read.
        column_names (list[str] | None): Optional column names to assign to the DataFrame.

    Returns:
        pd.DataFrame: The contents of the .oftable file as a pandas DataFrame.
    """
    table = pd.read_csv(
        filepath, comment="#", sep=r"[()\s]+", engine="python", header=None
    )
    # Remove empty columns
    table = table.dropna(axis=1, how="all")
    column_headers = extract_column_names(filepath)
    if column_names is not None:
        column_headers = column_names
        if len(column_names) != len(table.columns):
            column_names = None
    if column_headers is None or len(column_headers) != len(table.columns):
        column_headers = None
    update_column_names(table, column_headers)
    return table


@TableReader.register(".dat")
def read_dat(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a .dat file and return a DataFrame."""
    return read_oftable(filepath, column_names=column_names)


@TableReader.register(".raw")
def read_raw(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a .raw file and return a DataFrame."""
    if column_names is None:
        column_names = extract_column_names(filepath)
    table = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    update_column_names(table, column_names)
    return table


@TableReader.register("")
def read_default(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a file with no extension and return a DataFrame."""
    return read_oftable(filepath, column_names=column_names)


@TableReader.register(".xy")
def read_xy(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a .xy file and return a DataFrame."""
    if column_names is None:
        column_names = extract_column_names(filepath)
    table = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    update_column_names(table, column_names)
    return table


@TableReader.register(".csv")
def read_csv(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a .csv file and return a DataFrame."""
    with Path(filepath).open() as f:
        first_lines = list(islice(f, 20))

    non_comment_lines = [line for line in first_lines if not line.startswith("#")]

    # check if all of the lines can be converted to floats
    # assume the they are comma separated
    entries = [line.split(",") for line in non_comment_lines]

    # Check if all entries in each line are convertible to floats
    has_header = not all(is_convertible_to_float(entry) for entry in entries)

    if has_header:
        table = pd.read_csv(filepath, comment="#")
    else:
        table = pd.read_csv(filepath, comment="#", header=None)
    update_column_names(table, column_names)
    return table


def read_catch2_benchmark(
    filepath: str | Path, column_names: list[str] | None = None
) -> pd.DataFrame:
    """Read a Catch2 XML benchmark results file and return a DataFrame."""
    tree = parse(filepath)
    root = tree.getroot()
    if root is None:
        err_msg = f"Unable to parse XML file: {filepath}"
        raise ValueError(err_msg)

    records = []

    def _parse_sections(
        sections: list[ET.Element], test_case_name: str, section_path: list[str]
    ) -> None:
        for section in sections:
            name = section.attrib.get("name", "")
            new_path = [*section_path, name]

            subsections = section.findall("Section")
            if subsections:
                _parse_sections(subsections, test_case_name, new_path)
            elif (benchmark := section.find("BenchmarkResults")) is not None and (
                mean := benchmark.find("mean")
            ) is not None:
                record = {
                    "test_case": test_case_name,
                    "benchmark_name": benchmark.attrib.get("name"),
                    "avg_runtime": float(mean.attrib.get("value", 0)),
                }

                # Add dynamic section depth fields
                for i, sec_name in enumerate(new_path):
                    record[f"section{i + 1}"] = sec_name

                records.append(record)

    for testcase in root.findall("TestCase"):
        test_case_name = testcase.attrib.get("name")
        if test_case_name:
            _parse_sections(testcase.findall("Section"), str(test_case_name), [])

    table = pd.DataFrame(records)

    # Fill missing sectionN columns with empty string (not NaN)
    max_sections = max((len(r) - 18 for r in records), default=0)
    for i in range(1, max_sections + 1):
        col = f"section{i}"
        if col not in table.columns:
            table[col] = ""

    if column_names:
        table = table[column_names]
    return table  # ty: ignore[invalid-return-type]
