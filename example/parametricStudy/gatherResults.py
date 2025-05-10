#%%
import os
from pathlib import Path
from foamlib import FoamCase, FoamFile
from foamlib.postprocessing.load_tables import load_tables,OutputFile, list_outputfiles
import pandas as pd
import json

from collections import defaultdict


# damBreak
root = Path(__file__).parent
results = root / "results"
results.mkdir(exist_ok=True)
# %%
out_files = list_outputfiles(root / "Cases")

def max_height_filter(table: pd.DataFrame,parameters: dict[str,str]) -> pd.DataFrame:
    d = {
        "x": [table["x"].max()],
        "y": [table["y"].max()],
        "z": [table["z"].max()],
    }
    d.update(parameters)
    return pd.DataFrame(d)

# %%
forces = load_tables(
    output_file=out_files["forces--force.dat"],
    dir_name=root / "Cases"
)
# %%
forces.to_csv(
    results / "forces.csv",
    index=False,
)

probeU = load_tables(
    output_file=out_files["probes--U"],
    dir_name=root / "Cases"
)
probeU.to_csv(
    results / "probeU.csv",
    index=False,
)

file = OutputFile(file_name="U_freeSurface.raw", folder="freeSurface")
surfaceHeights = load_tables(
    output_file=file, dir_name=root / "Cases", filter=max_height_filter
)
surfaceHeights.to_csv(
    results / "surfaceHeights.csv",
    index=False,
)



# %%
