# %%
import json
from pathlib import Path

import pandas as pd

from foamlib import FoamCase, FoamFile

# damBreak
root = Path(__file__).parent

parastudy = pd.read_csv(root / "parastudy.csv").to_dict(orient="records")
# %%
for case in parastudy:
    case_dir = root / "Cases" / case["case_name"]
    ofcase = FoamCase(root / "damBreak")
    ofcase.clone(case_dir)

    parameter = FoamFile(case_dir / "system" / "simulationsParameters").as_dict()
    parameter_keys = set(parameter.keys())
    case_keys = set(case.keys())
    case_only_keys = case_keys - parameter_keys

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["res1"] = case["res1"]

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["res2"] = case["res2"]

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["res3"] = case["res3"]

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["res4"] = case["res4"]

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["res5"] = case["res5"]

    with FoamFile(case_dir / "system" / "simulationsParameters") as f:
        f["fluidHeight"] = case["fluidHeight"]

    d = {"parameters": {}}
    for key in case_only_keys:
        if key == "case_name":
            continue
        d["parameters"][key] = case[key]
    with open(case_dir / "parameters.json", "w") as f:
        json.dump(d, f, indent=4)
