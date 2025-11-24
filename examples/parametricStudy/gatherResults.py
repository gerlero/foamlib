# %%
from pathlib import Path

import pandas as pd

from foamlib.postprocessing.load_tables import DataSource, list_function_objects, load_tables

# damBreak
root = Path(__file__).parent
results = root / "results"
results.mkdir(exist_ok=True)
# %%
out_files = list_function_objects(root / "Cases")


def max_height_filter(table: pd.DataFrame, parameters: list[dict[str, str]]) -> pd.DataFrame:
    """Filter the table to get the maximum height."""
    d = {
        "x": [table["x"].max()],
        "y": [table["y"].max()],
        "z": [table["z"].max()],
    }
    d.update(parameters)  # ty: ignore[no-matching-overload]
    return pd.DataFrame(d)


# %%
forces = load_tables(
    source=out_files["forces--force.dat"], dir_name=root / "Cases"
)
# %%
assert forces is not None
forces.to_csv(
    results / "forces.csv",
    index=False,
)

probe_u = load_tables(source=out_files["probes--U"], dir_name=root / "Cases")
assert probe_u is not None
probe_u.to_csv(
    results / "probe_u.csv",
    index=False,
)

file = DataSource(file_name="U_freeSurface.raw", folder="freeSurface")  # ty: ignore[missing-argument]
surface_heights = load_tables(
    source=file, dir_name=root / "Cases", filter_table=max_height_filter
)
assert surface_heights is not None
surface_heights.to_csv(
    results / "surface_heights.csv",
    index=False,
)


# %%
