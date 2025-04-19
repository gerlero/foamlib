from __future__ import annotations

import sys
from enum import Enum
from typing import Dict, NamedTuple, Optional, Union

import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, MutableMapping, Sequence
else:
    from typing import Mapping, MutableMapping, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


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
    "np.ndarray[tuple[int], np.dtype[np.float64]]",
]

TensorLike = Union[
    Sequence[float],
    "np.ndarray[tuple[()], np.dtype[np.float64]]",
    Tensor,
]


class TensorKind(Enum):
    SCALAR = ()
    VECTOR = (3,)
    SYMM_TENSOR = (6,)
    TENSOR = (9,)

    @property
    def shape(self) -> tuple[()] | tuple[int]:
        shape: tuple[()] | tuple[int] = self.value
        return shape

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __str__(self) -> str:
        return {
            TensorKind.SCALAR: "scalar",
            TensorKind.VECTOR: "vector",
            TensorKind.SYMM_TENSOR: "symmTensor",
            TensorKind.TENSOR: "tensor",
        }[self]

    @staticmethod
    def from_shape(shape: tuple[int, ...]) -> TensorKind:
        for kind in TensorKind:
            if kind.shape == shape:
                return kind

        msg = f"No tensor kind for shape {shape!r}"
        raise ValueError(msg)


class Dimensioned:
    def __init__(
        self,
        value: TensorLike,
        dimensions: DimensionSet | Sequence[float],
        name: str | None = None,
    ) -> None:
        if is_sequence(value):
            self.value: Tensor = np.array(value, dtype=float)  # type: ignore [assignment]
        else:
            assert isinstance(value, (int, float, np.ndarray))
            self.value = float(value)

        if not isinstance(dimensions, DimensionSet):
            self.dimensions = DimensionSet(*dimensions)
        else:
            self.dimensions = dimensions

        self.name = name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimensioned):
            return NotImplemented

        return (
            self.dimensions == other.dimensions
            and np.array_equal(self.value, other.value)
            and self.name == other.name
        )


Field = Union[
    float,
    "np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]",
]

FieldLike = Union[
    TensorLike,
    Sequence[TensorLike],
    Sequence[Sequence[TensorLike]],
    Field,
]


Data = Union[
    str,
    int,
    float,
    bool,
    Dimensioned,
    DimensionSet,
    Sequence["Entry"],
    Field,
]

Entry = Union[
    Data,
    Mapping[str, "Entry"],
]
"""
A value that can be stored in an OpenFOAM file.
"""

DataLike = Union[
    FieldLike,
    Sequence["EntryLike"],
    Data,
]

EntryLike = Union[
    DataLike,
    Mapping[str, "EntryLike"],
]


def is_sequence(
    value: EntryLike,
) -> TypeGuard[
    Sequence[EntryLike]
    | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]
]:
    return (isinstance(value, Sequence) and not isinstance(value, str)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    )


MutableEntry = Union[
    Data,
    MutableMapping[str, "MutableEntry"],
]

Dict_ = Dict[str, Union["Entry", "Dict_"]]
File = Dict[Optional[str], Union["Entry", "Dict_"]]
