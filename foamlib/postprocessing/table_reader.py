"""This module provides a utility class for reading tabular data from files with various extensions."""

from __future__ import annotations  # Add this import at the top of the file

from itertools import islice
from pathlib import Path
from typing import Callable, ClassVar, Optional, Union

import pandas as pd


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
        dict[str, Callable[[Union[str, Path], Optional[list[str]]], pd.DataFrame]]
    ] = {}

    def __init__(self) -> None:
        """Initialize the TableReader instance."""

    @classmethod
    def register(cls, extension: str) -> Callable[
        [Callable[[Union[str, Path], Optional[list[str]]], pd.DataFrame]],
        Callable[[Union[str, Path], Optional[list[str]]], pd.DataFrame],
    ]:
        """
        Register a reader function for a specific file extension.

        The extension is case-insensitive.

        Args:
            extension (str): The file extension (e.g., ".dat", ".raw") to register the reader for.

        Returns:
            Callable[[Callable[[str, Optional[list[str]]], pd.DataFrame]], Callable[[str, Optional[list[str]]], pd.DataFrame]]:
                A decorator that registers the function as a reader for the specified extension.
        """

        def decorator(
            func: Callable[[Union[str, Path], Optional[list[str]]], pd.DataFrame],
        ) -> Callable[[Union[str, Path], Optional[list[str]]], pd.DataFrame]:
            cls._registry[extension.lower()] = func
            return func

        return decorator

    def read(
        self, filepath: Union[str, Path], column_names: Optional[list[str]] = None
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
        ext = str(Path(filepath).suffix.lower())
        if ext not in self._registry:
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


def extract_column_names(filepath: Union[str, Path]) -> Optional[list[str]]:
    """
    Extract column names from the first 20 lines of a file.

    Args:
        filepath (str): The path to the file from which to extract column names.
    Returns:
        Optional[list[str]]: A list of column names extracted from the file, or None if no
            comment lines are found.
    """
    with open(filepath) as f:
        first_lines = [line.strip() for line in islice(f, 20)]

    # Filter only comment lines
    comment_lines = [line for line in first_lines if line.startswith("#")]

    if not comment_lines:
        return None

    # Take the last comment line and split into column names
    last_comment = comment_lines[-1]
    headers = last_comment.lstrip("#").strip()
    return headers.split()


def update_column_names(
    table: pd.DataFrame, column_names: Optional[list[str]]
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
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Use a regular expression to parse the file and separate on parentheses and whitespace.

    Args:
        filepath (str): The path to the .oftable file to be read.
        column_names (Optional[list[str]]): Optional column names to assign to the DataFrame.

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
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a .dat file and return a DataFrame."""
    return read_oftable(filepath, column_names=column_names)


@TableReader.register(".raw")
def read_raw(
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a .raw file and return a DataFrame."""
    if column_names is None:
        column_names = extract_column_names(filepath)
    table = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    update_column_names(table, column_names)
    return table


@TableReader.register("")
def read_default(
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a file with no extension and return a DataFrame."""
    return read_oftable(filepath, column_names=column_names)


@TableReader.register(".xy")
def read_xy(
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a .xy file and return a DataFrame."""
    if column_names is None:
        column_names = extract_column_names(filepath)
    table = pd.read_csv(filepath, comment="#", sep=r"\s+", header=None)
    update_column_names(table, column_names)
    return table


@TableReader.register(".csv")
def read_csv(
    filepath: Union[str, Path], column_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read a .csv file and return a DataFrame."""
    with open(filepath) as f:
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
