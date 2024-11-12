from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, MutableMapping, Sequence
else:
    from typing import Mapping, MutableMapping, Sequence


class DimensionSet(NamedTuple):
    mass: float = 0
    length: float = 0
    time: float = 0
    temperature: float = 0
    moles: float = 0
    current: float = 0
    luminous_intensity: float = 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self) if v != 0)})"


Tensor = Union[
    float,
    Sequence[float],
    "np.ndarray[Tuple[()], np.dtype[np.generic]]",
    "np.ndarray[Tuple[int], np.dtype[np.generic]]",
]


@dataclass
class Dimensioned:
    value: Tensor = 0
    dimensions: DimensionSet | Sequence[float] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, DimensionSet):
            self.dimensions = DimensionSet(*self.dimensions)


Field = Union[
    Tensor, Sequence[Tensor], "np.ndarray[Tuple[int, int], np.dtype[np.generic]]"
]

DataEntry = Union[
    str,
    int,
    float,
    bool,
    Dimensioned,
    DimensionSet,
    Sequence["Data"],
    Tensor,
    Field,
]

Data = Union[
    DataEntry,
    Mapping[str, "Data"],
]
"""
A value that can be stored in an OpenFOAM file.
"""

MutableData = Union[
    DataEntry,
    MutableMapping[str, "MutableData"],
]

Dict_ = Dict[str, Union["Data", "Dict_"]]
File = Dict[Optional[str], Union["Data", "Dict_"]]
