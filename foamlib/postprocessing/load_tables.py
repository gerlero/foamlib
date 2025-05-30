"""Load OpenFOAM post-processing tables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

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


def of_cases(dir_name: Union[str, Path]) -> list[str]:
    """List all OpenFOAM cases in folder.

    Parameters
    ----------
    dir_name : str
        name of the search directory

    Returns
    ofcases : List[str]
        pathes of the OpenFOAM directories
    """
    cases = []
    for path, dirnames, _ in os.walk(dir_name):
        if _of_case(dirnames):
            cases.append(path)
            dirnames[:] = []
    return cases


@dataclass
class OutputFile:
    """Class to represent an output file in OpenFOAM post-processing.

    Attributes
    ----------
    file_name : str
        Name of the output file.
    folder : str or Path
        Path to the folder containing the output file.
    _times : set[str]
        Set of time steps for the output file.
    """

    file_name: str
    folder: Union[str, Path]
    _times: set[str] = field(default_factory=set, init=False, repr=False)

    def add_time(self, t: str) -> None:
        """Add a time step to the output file.

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


def load_tables(
    output_file: OutputFile,
    dir_name: Union[str, Path],
    filter_table: Optional[
        Callable[[pd.DataFrame, list[dict[str, str]]], pd.DataFrame]
    ] = None,
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all available dataframes for an OutputFile across time steps.

    Parameters
    ----------
    output_file : OutputFile
        OutputFile object containing file name, time steps, and folder
    dir_name : Union[str, Path]
        Root directory where OpenFOAM cases are stored

    Returns
    -------
    pd.DataFrame or None
        Concatenated dataframe of all available time steps, or None if nothing found
    """
    all_tables = []

    for case in of_cases(dir_name):
        case_path = Path(case)
        postproc_root = case_path / "postProcessing"

        if not postproc_root.exists():
            continue

        postproc_folder = postproc_root / output_file.folder

        if not output_file.times:
            for time in postproc_folder.iterdir():
                if time.is_dir() and _is_float(time.name):
                    output_file.add_time(time.name)

        for time_value in output_file.times:
            file_path = postproc_folder / time_value / output_file.file_name

            if file_path.exists():
                reader = TableReader()
                table = reader.read(file_path)
                json_path = Path(case_path) / "case.json"

                with open(json_path) as f:
                    json_data = json.load(f)
                    parameters = json_data.get("case_parameters", {})
                    if len(output_file.times) > 1:
                        parameters.append({"category": "timeValue", "name": time_value})

                    # add parameters as columns
                    for parameter in parameters:
                        category = parameter["category"]
                        name = parameter["name"]
                        table[category] = name
                if filter_table is not None:
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


def _outputfiles(file_map: dict[str, OutputFile], postproc_root: Path) -> None:
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
                file_map[key] = OutputFile(file_name=fname, folder=folder)

            file_map[key].add_time(time)


def list_outputfiles(cases_folder: str = "Cases") -> dict[str, OutputFile]:
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
    file_map: dict[str, OutputFile] = {}

    for case_path_str in of_cases(cases_folder):
        case_path = Path(case_path_str)
        postproc_root = case_path / "postProcessing"
        if not postproc_root.exists():
            continue

        _outputfiles(file_map, postproc_root)

    return file_map
