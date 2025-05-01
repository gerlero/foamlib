from __future__ import annotations  # Add this import at the top of the file

import os


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
    return "constant" in dirnames and "system" in dirnames



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
