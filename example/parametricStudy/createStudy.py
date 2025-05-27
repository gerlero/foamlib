# %%
import json
from pathlib import Path

import pandas as pd

from foamlib import FoamCase, FoamFile
from foamlib.preprocessing.parameter_study import csv_generator

# damBreak
root = Path(__file__).parent
template_case = root / "damBreak" 

study = csv_generator(
    csv_file=root / "parastudy.csv", template_case=template_case, output_folder=root / "Cases"
)

study.create_study()