from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Sequence, Union


FoamDimensionSet = namedtuple(
    "FoamDimensionSet",
    [
        "mass",
        "length",
        "time",
        "temperature",
        "moles",
        "current",
        "luminous_intensity",
    ],
    defaults=(0, 0, 0, 0, 0, 0, 0),
)


@dataclass
class FoamDimensioned:
    value: Union[int, float, Sequence[Union[int, float]]] = 0
    dimensions: Union[FoamDimensionSet, Sequence[Union[int, float]]] = (
        FoamDimensionSet()
    )
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, FoamDimensionSet):
            self.dimensions = FoamDimensionSet(*self.dimensions)


FoamValue = Union[
    str, int, float, bool, FoamDimensioned, FoamDimensionSet, Sequence["FoamValue"]
]
"""
A value that can be stored in an OpenFOAM dictionary.
"""
