from itertools import islice
from pathlib import Path
from typing import Callable, Dict,Optional, List

import pandas as pd

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

        read(filepath: str) -> pd.DataFrame:
            Reads a file and returns its contents as a pandas DataFrame. The file extension is used
            to determine the appropriate reader function. Raises a ValueError if no reader is registered
            for the file's extension.
    """

    _registry: Dict[str, Callable[[str], pd.DataFrame]] = {}

    def __init__(self):
        pass

    @classmethod
    def register(cls, extension: str):
        """
        A class method decorator used to register a reader function for a specific file extension.
        The extension is case-insensitive.
        Args:
            extension (str): The file extension (e.g., ".dat", ".raw") to register the reader for.
        Returns:
            Callable[[Callable[[str], pd.DataFrame]], Callable[[str], pd.DataFrame]]:
                A decorator that registers the function as a reader for the specified extension.
        """

        def decorator(func: Callable[[str], pd.DataFrame]):
            cls._registry[extension.lower()] = func
            return func

        return decorator

    def read(self, filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reads a file and returns its contents as a pandas DataFrame.
        The file extension is used to determine the appropriate reader function.
        Raises:
            ValueError: If no reader is registered for the file's extension.
        Args:
            filepath (str): The path to the file to be read.
        Returns:
            pd.DataFrame: The contents of the file as a pandas DataFrame.
        """
        ext = Path(filepath).suffix.lower()
        if ext not in self._registry:
            raise ValueError(f"No reader registered for extension: '{ext}'")
        return self._registry[ext](filepath, column_names=column_names)


def extract_column_names(filepath: str) -> Optional[List[str]]:
    with open(filepath, "r") as f:
        first_lines = [line.strip() for line in islice(f, 20)]

    # Filter only comment lines
    comment_lines = [line for line in first_lines if line.startswith("#")]
    
    if not comment_lines:
        return None
    
    # Take the last comment line and split into column names
    last_comment = comment_lines[-1]
    headers = last_comment.lstrip("#").strip()
    return headers.split()


def update_column_names(df: pd.DataFrame, column_names: Optional[List[str]]) -> pd.DataFrame:
    """
    Update the column names of a DataFrame if provided.
    Args:
        df (pd.DataFrame): The DataFrame to update.
        column_names (Optional[List[str]]): The new column names to set.
    Returns:
        pd.DataFrame: The updated DataFrame with new column names.
    """

    if column_names is not None:
        if len(column_names) != len(df.columns):
            raise ValueError(f"Number of column names ({len(column_names)}) does not match number of columns in DataFrame ({df.columns}).")
        df.columns = column_names
    return df


def read_oftable(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    This function uses a regular expression to parse the file and seperates on
    paratheses and whitespace
    Args:
        filepath (str): The path to the .oftable file to be read.
    Returns:
        pd.DataFrame: The contents of the .oftable file as a pandas DataFrame.
    """
    df = pd.read_csv(filepath, comment="#", sep="[()\s]+", engine="python", header=None)
    # Remove empty columns
    df = df.dropna(axis=1, how="all")
    update_column_names(df, column_names)
    return df


@TableReader.register(".dat")
def read_dat(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a .dat file and return a DataFrame."""

    return read_oftable(filepath,column_names=column_names)


@TableReader.register(".raw")
def read_raw(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a .raw file and return a DataFrame."""

    df = pd.read_csv(filepath, comment="#", sep="\s+", header=None)
    update_column_names(df, column_names)
    return df


@TableReader.register("")
def read_default(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a file with no extension and return a DataFrame."""

    return read_oftable(filepath,column_names=column_names)


@TableReader.register(".xy")
def read_xy(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a .xy file and return a DataFrame."""

    df = pd.read_csv(filepath, comment="#", sep="\s+", header=None)
    update_column_names(df, column_names)
    return df


@TableReader.register(".csv")
def read_csv(filepath: str, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a .csv file and return a DataFrame."""

    df = pd.read_csv(filepath, comment="#", header=None)
    update_column_names(df, column_names)
    return df
