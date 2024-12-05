from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Union

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
    "np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.int_]]",
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
    Tensor,
    Sequence[Tensor],
    "np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.int_]]",
]

Data = Union[
    str,
    int,
    float,
    bool,
    Dimensioned,
    DimensionSet,
    Sequence["Entry"],
    Tensor,
    Field,
]

Entry = Union[
    Data,
    Mapping[str, "Entry"],
]
"""
A value that can be stored in an OpenFOAM file.
"""

MutableEntry = Union[
    Data,
    MutableMapping[str, "MutableEntry"],
]

Dict_ = Dict[str, Union["Entry", "Dict_"]]
File = Dict[Optional[str], Union["Entry", "Dict_"]]
