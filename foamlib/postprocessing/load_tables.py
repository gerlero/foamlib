from __future__ import annotations  # Add this import at the top of the file

from dataclasses import dataclass, field
import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from collections import defaultdict
from .table_reader import TableReader


def _of_case(dirnames: list[str], filenames: list[str]) -> bool:
    """Classify directory as OpenFOAM case

    Parameters
    ----------
    dirnames : list[str]
        list of directories in the folder
    filenames : list[str]
        list of files in the folder

    Returns
    ofcase : bool
        is the folder an OpenFOAM Case
    """
    hasConstant = "constant" in dirnames
    hasSystem = "system" in dirnames
    return hasConstant and hasSystem


def of_cases(dir_name: str) -> list[str]:
    """List all OpenFOAM cases in folder

    Parameters
    ----------
    dir_name : str
        name of the search directory

    Returns
    ofcases : List[str]
        pathes of the OpenFOAM directories
    """
    cases = []
    for path, dirnames, filenames in os.walk(dir_name):
        if _of_case(dirnames, filenames):
            cases.append(path)
            dirnames[:] = []
    return cases


@dataclass
class OutputFile:
    file_name: str
    folder: Union[str, Path] = None
    _times: set[str] = field(default_factory=set, init=False, repr=False)

    def add_time(self, t: str):
        self._times.add(t)

    @property
    def times(self) -> list[str]:
        return sorted(self._times)


def load_tables(
    output_file: OutputFile, dir_name: str, filter: Optional[callable[[pd.DataFrame, dict[str, str]], pd.DataFrame]]=None
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all available dataframes for an OutputFile across time steps.

    Parameters
    ----------
    output_file : OutputFile
        OutputFile object containing file name, time steps, and folder
    dir_name : str
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
            [
                output_file.add_time(f.name)
                for f in postproc_folder.iterdir()
                if f.is_dir()
            ]

        for time in output_file.times:
            file_path = postproc_folder / time / output_file.file_name

            if file_path.exists():
                print(f"Loading {file_path}")
                reader = TableReader()
                table = reader.read(file_path)
                json_path = Path(case_path) / "parameters.json"

                with open(json_path) as f:
                    json_data = json.load(f)
                    parameters = json_data.get("parameters", {})
                    if len(output_file.times) > 1:
                        parameters["timeValue"] = float(time)
                    # add parameters as columns
                    for key, value in parameters.items():
                        table[key] = value
                if filter is not None:
                    table = filter(table,parameters)
                all_tables.append(table)

    if all_tables:
        return pd.concat(all_tables, ignore_index=True)
    else:
        print(f"No data found for {output_file.file_name}")
        return None


def is_float(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _outputfiles(file_map, case_path_str, postproc_root):
    for dirpath, _, filenames in os.walk(postproc_root):
        base = os.path.basename(dirpath)

        if not is_float(base):
            # Skip directories that are not time directories
            continue

        time = base
        time_path = Path(dirpath)
        rel_to_postproc = time_path.relative_to(postproc_root)
        folder = rel_to_postproc.parent
        folder_str = str(folder) if folder != Path(".") else ""

        for fname in filenames:
            key = f"{folder_str}--{fname}"
            full_folder = os.path.join(case_path_str, "postProcessing", folder)

            if key not in file_map:
                file_map[key] = OutputFile(file_name=fname, folder=full_folder)

            file_map[key].add_time(time)


def list_outputfiles(Cases: str = "Cases") -> dict[str, list[OutputFile]]:
    file_map: dict[str, OutputFile] = {}

    for case_path_str in of_cases(Cases):
        case_path = Path(case_path_str)
        postproc_root = case_path / "postProcessing"
        if not postproc_root.exists():
            continue

        _outputfiles(file_map, case_path_str, postproc_root)

    return file_map
