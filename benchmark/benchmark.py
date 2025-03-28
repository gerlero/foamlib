#!/usr/bin/env python3

"""Benchmark foamlib against PyFoam."""

import timeit

from foamlib import FoamFieldFile
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

FoamFieldFile("U").internal_field = [[0.0, 0.0, 0.0]] * 200_000

print(
    f"foamlib: {min(timeit.repeat(lambda: FoamFieldFile('U').internal_field, number=1))} s"
)
print(
    f"PyFoam: {min(timeit.repeat(lambda: ParsedParameterFile('U')['internalField'], number=1))} s"
)
