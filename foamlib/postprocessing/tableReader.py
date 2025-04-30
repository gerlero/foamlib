import pandas as pd
from pathlib import Path
from typing import Callable, Dict


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

    def read(self, filepath: str) -> pd.DataFrame:
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
        return self._registry[ext](filepath)


def read_oftable(filepath: str) -> pd.DataFrame:
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
    return df


@TableReader.register(".dat")
def read_dat(filepath: str) -> pd.DataFrame:
    """Read a .dat file and return a DataFrame."""

    return read_oftable(filepath)


@TableReader.register(".raw")
def read_raw(filepath: str) -> pd.DataFrame:
    """Read a .raw file and return a DataFrame."""

    return pd.read_csv(filepath, comment="#", sep="\s+", header=None)


@TableReader.register("")
def read_default(filepath: str) -> pd.DataFrame:
    """Read a file with no extension and return a DataFrame."""

    return read_oftable(filepath)


@TableReader.register(".xy")
def read_xy(filepath: str) -> pd.DataFrame:
    """Read a .xy file and return a DataFrame."""

    return pd.read_csv(filepath, comment="#", sep="\s+", header=None)


@TableReader.register(".csv")
def read_csv(filepath: str) -> pd.DataFrame:
    """Read a .csv file and return a DataFrame."""

    return pd.read_csv(filepath, comment="#", header=None)
