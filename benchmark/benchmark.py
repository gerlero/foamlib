#!/usr/bin/env python3

"""Benchmark foamlib against PyFoam."""

import timeit

from foamlib import FoamFieldFile
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

FoamFieldFile("U").internal_field = [[0.0, 0.0, 0.0]] * 1_000_000
with FoamFieldFile("U_binary") as f:
    f.format = "binary"
    f.internal_field = [[0.0, 0.0, 0.0]] * 1_000_000

print(
    f"foamlib (binary): {min(timeit.repeat(lambda: FoamFieldFile('U_binary').internal_field, number=1))} s"
)
print(
    f"foamlib (ASCII): {min(timeit.repeat(lambda: FoamFieldFile('U').internal_field, number=1))} s"
)
print(
    f"PyFoam (binary): {min(timeit.repeat(lambda: ParsedParameterFile('U_binary')[
                'internalField'
            ], number=1))} s"
)
print(
    f"PyFoam (ASCII): {min(timeit.repeat(lambda: ParsedParameterFile('U')[
                'internalField'
            ], number=1))} s"
)
