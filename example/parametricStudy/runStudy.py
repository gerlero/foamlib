#%%
import os
from pathlib import Path
from foamlib import FoamCase
from foamlib.postprocessing.load_tables import of_cases

root = Path(__file__).parent

for case in of_cases(root / "Cases"):
    print(case)
    of_case = FoamCase(case)
    of_case.run()
# %%
