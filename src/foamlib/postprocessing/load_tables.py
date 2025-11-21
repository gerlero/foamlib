"""Load OpenFOAM post-processing tables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import pandas as pd
except ImportError as e:
    msg = "The postprocessing module requires extra dependencies. Install 'foamlib[postprocessing]' to use it."
    raise ImportError(msg) from e

from .table_reader import TableReader


def _of_case(dirnames: list[str]) -> bool:
    """Classify directory as OpenFOAM case.

    Parameters
    ----------
    dirnames : list[str]
        list of directories in the folder

    Returns
    ofcase : bool
        is the folder an OpenFOAM Case
    """
    has_constant = "constant" in dirnames
    has_system = "system" in dirnames
    return has_constant and has_system


def of_cases(dir_name: str | Path) -> list[str]:
    """List all OpenFOAM cases in folder.

    Parameters
    ----------
    dir_name : str
        name of the search directory

    Returns
    ofcases : List[str]
        paths of the OpenFOAM directories
    """
    cases = []
    for path, dirnames, _ in os.walk(dir_name):
        if _of_case(dirnames):
            cases.append(path)
            dirnames[:] = []
    return cases


@dataclass
class DataSource:
    """
    Describes a location of simulation output data inside a case directory.

    Attributes
    ----------
    file_name : str
        The name of the file to be read (e.g., 'forces.dat').
    folder : Union[str, Path]
        The subdirectory where the file is located, relative to case path.
    time_resolved : bool
        Whether data is stored in time-specific subdirectories.
    postproc_prefix : str
        Prefix for the post-processing directory, typically 'postProcessing'.
    """

    file_name: str
    folder: str | Path
    postproc_prefix: str
    time_resolved: bool = True
    _times: set[str] = field(default_factory=set, init=False, repr=False)

    def add_time(self, t: str) -> None:
        """Add a time step to the data source.

        Parameters
        ----------
        t : str
            Time step to add.
        """
        self._times.add(t)

    @property
    def times(self) -> list[str]:
        """Get the list of time steps for this output file.

        Returns
        -------
        list[str]
            List of time steps as strings.
        """
        return sorted(self._times)

    def postproc_folder(self, case_path: Path) -> Path:
        """Return the path to the target's base directory under postProcessing/."""
        return case_path / self.postproc_prefix / self.folder

    def resolve_paths(self, case_path: Path) -> list[Path]:
        """
        Compute full file paths for this target inside the given case.

        Parameters
        ----------
        case_path : Path
            Root path of the case directory.

        Returns
        -------
        list[Path]
            List of resolved file paths to load.
        """
        base = self.postproc_folder(case_path)
        if self.time_resolved:
            return [base / t / self.file_name for t in self.times]
        return [base / self.file_name]


def functionobject(file_name: str, folder: str | Path) -> DataSource:
    """
    Create a DataSource for a standard OpenFOAM function object.

    Parameters
    ----------
    file_name : str
        The file name to look for.
    folder : str | Path
        The function object name (and folder).

    Returns
    -------
    DataSource
        A DataSource object configured for the function object.
    """
    return DataSource(
        file_name=file_name,
        folder=folder,
        time_resolved=True,
        postproc_prefix="postProcessing",
    )


def datafile(
    file_name: str, folder: str | Path, *, time_resolved: bool = False
) -> DataSource:
    """
    Create a DataSource for a custom or non-OpenFOAM output file.

    Parameters
    ----------
    file_name : str
        Name of the file (e.g., 'output.xml').
    folder : str or Path
        Subdirectory where the file is located (relative to 'postProcessing/').
    time_resolved : bool
        Whether the data is organized by time subfolders.

    Returns
    -------
    DataSource
        A DataSource object configured for the custom file.
    """
    return DataSource(
        file_name=file_name,
        folder=folder,
        time_resolved=time_resolved,
        postproc_prefix=".",
    )


def load_tables(
    source: DataSource,
    dir_name: str | Path,
    filter_table: Callable[[pd.DataFrame, list[dict[str, str]]], pd.DataFrame]
    | None = None,
    reader_fn: Callable[[Path], pd.DataFrame | None] | None = None,
) -> pd.DataFrame | None:
    """
    Load and concatenate all available dataframes for a DataTarget across cases and time steps.

    Parameters
    ----------
    source : DataSource
        source data descriptor for resolving output paths.
    dir_name : str or Path
        Root directory where OpenFOAM cases are stored.
    filter_table : callable, optional
        Function to filter or modify the dataframe after reading.
    reader_fn : callable, optional
        Function to read a file into a DataFrame. Defaults to TableReader().read.
        that considers the specific file format

    Returns
    -------
    pd.DataFrame or None
        Concatenated dataframe of all found data, or None if nothing was found.
    """
    all_tables = []
    reader_fn = reader_fn or TableReader().read

    for case in of_cases(dir_name):
        case_path = Path(case)
        target_folder = source.postproc_folder(case_path)

        # Skip if the target folder does not exist
        if not target_folder.exists():
            continue

        # Discover time steps if needed
        if source.time_resolved and not source.times:
            for item in target_folder.iterdir():
                if item.is_dir() and _is_float(item.name):
                    source.add_time(item.name)

        for file_path in source.resolve_paths(case_path):
            if not file_path.exists():
                continue

            table = reader_fn(file_path)
            if table is None:
                continue

            # Load case metadata
            json_path = case_path / "case.json"
            parameters = []
            if json_path.exists():
                with json_path.open() as f:
                    json_data = json.load(f)
                    parameters = json_data.get("case_parameters", [])

            # Add time value if applicable
            if source.time_resolved and len(source.times) > 1:
                time_str = file_path.parent.name
                parameters.append({"category": "timeValue", "name": time_str})

            # add parameters as columns
            for parameter in parameters:
                category = parameter["category"]
                name = parameter["name"]
                table[category] = name

            if filter_table:
                table = filter_table(table, parameters)

            all_tables.append(table)

    if all_tables:
        return pd.concat(all_tables, ignore_index=True)

    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _discover_function_objects(
    file_map: dict[str, DataSource], postproc_root: Path
) -> None:
    for dirpath, _, filenames in os.walk(postproc_root):
        base = Path(dirpath).name

        if not _is_float(base):
            # Skip directories that are not time directories
            continue

        time = base
        time_path = Path(dirpath)
        rel_to_postproc = time_path.relative_to(postproc_root)
        folder = rel_to_postproc.parent
        folder_str = str(folder) if folder != Path() else ""

        for fname in filenames:
            key = f"{folder_str}--{fname}"

            if key not in file_map:
                file_map[key] = functionobject(file_name=fname, folder=folder)

            file_map[key].add_time(time)


def list_function_objects(cases_folder: str | Path = "Cases") -> dict[str, DataSource]:
    """List all output files in OpenFOAM cases.

    Parameters
    ----------
    cases_folder : str
        Name of the search directory.

    Returns
    -------
    file_map : dict[str, list[OutputFile]]
        Dictionary with keys as file names and values as OutputFile objects.
    """
    file_map: dict[str, DataSource] = {}

    for case_path_str in of_cases(cases_folder):
        case_path = Path(case_path_str)
        postproc_root = case_path / "postProcessing"
        if not postproc_root.exists():
            continue

        _discover_function_objects(file_map, postproc_root)

    return file_map
