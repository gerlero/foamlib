from __future__ import annotations

import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

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

    def __add__(self, other: DimensionSet) -> DimensionSet:  # type: ignore[override]
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot add DimensionSet with different dimensions: {self} + {other}"
            raise ValueError(msg)

        return self

    def __sub__(self, other: DimensionSet) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot subtract DimensionSet with different dimensions: {self} - {other}"
            raise ValueError(msg)

        return self

    def __mul__(self, other: DimensionSet) -> DimensionSet:  # type: ignore[override]
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a + b for a, b in zip(self, other)))

    def __truediv__(self, other: DimensionSet) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a - b for a, b in zip(self, other)))

    def __pow__(self, exponent: float) -> DimensionSet:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return DimensionSet(*(a * exponent for a in self))

    def __bool__(self) -> bool:
        return any(v != 0 for v in self)


Tensor = Union[
    float,
    "np.ndarray[tuple[int], np.dtype[np.float64]]",
]

TensorLike = Union[
    Tensor,
    Sequence[float],
]


class Dimensioned:
    def __init__(
        self,
        value: TensorLike,
        dimensions: DimensionSet | Sequence[float],
        name: str | None = None,
    ) -> None:
        if is_sequence(value):
            self.value: Tensor = np.array(value, dtype=float)
        else:
            assert isinstance(value, (int, float, np.ndarray))
            self.value = float(value)

        if not isinstance(dimensions, DimensionSet):
            self.dimensions = DimensionSet(*dimensions)
        else:
            self.dimensions = dimensions

        self.name = name

    def __repr__(self) -> str:
        if self.name is not None:
            return (
                f"{type(self).__name__}({self.value}, {self.dimensions}, {self.name})"
            )
        return f"{type(self).__name__}({self.value}, {self.dimensions})"

    def __add__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value + other.value,
            self.dimensions + other.dimensions,
            f"{self.name}+{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __sub__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value - other.value,
            self.dimensions - other.dimensions,
            f"{self.name}-{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __mul__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value * other.value,
            self.dimensions * other.dimensions,
            f"{self.name}*{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __truediv__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value / other.value,
            self.dimensions / other.dimensions,
            f"{self.name}/{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __pow__(self, exponent: float) -> Dimensioned:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return Dimensioned(
            self.value**exponent,
            self.dimensions**exponent,
            f"pow({self.name},{exponent})" if self.name is not None else None,
        )

    def __float__(self) -> float:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to float: {self.dimensions}"
            raise ValueError(msg)
        return float(self.value)

    def __int__(self) -> int:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to int: {self.dimensions}"
            raise ValueError(msg)
        return int(self.value)

    def __array__(
        self, dtype: Any = None, *, copy: Any = None
    ) -> np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64]]:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to array: {self.dimensions}"
            raise ValueError(msg)
        return np.array(self.value, dtype=dtype, copy=copy)


Field = Union[
    float,
    "np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]",
]

FieldLike = Union[
    Field,
    TensorLike,
    Sequence[TensorLike],
]

KeywordEntry = Tuple["DataEntry", Union["DataEntry", "SubDict"]]
KeywordEntryLike = Tuple["DataEntryLike", Union["DataEntryLike", "SubDictLike"]]

DataEntry = Union[
    str,
    int,
    float,
    bool,
    Dimensioned,
    DimensionSet,
    List[Union["DataEntry", KeywordEntry]],
    Field,
]
DataEntryLike = Union[
    DataEntry,
    Sequence[
        Union[
            "DataEntryLike",
            "KeywordEntryLike",
        ]
    ],
    FieldLike,
]

Data = Union[
    DataEntry,
    Tuple[DataEntry, ...],
]
DataLike = Union[
    DataEntryLike,
    Tuple["DataEntryLike", ...],
]

StandaloneData = Union[
    Data,
    "np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]",
    "np.ndarray[tuple[int, int], np.dtype[np.float64 | np.float32]]",
    List["np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]"],
    Tuple[
        "np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]",
        "np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]",
    ],
]
StandaloneDataLike = Union[
    StandaloneData,
    DataLike,
    Sequence["np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]"],
    Sequence[Sequence[int]],
    Tuple[Sequence[int], Sequence[int]],
]


def is_sequence(
    value: DataLike | StandaloneDataLike | SubDictLike,
) -> TypeGuard[
    Sequence[DataLike | tuple[DataLike, DataLike | SubDictLike]]
    | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]
]:
    return (isinstance(value, Sequence) and not isinstance(value, str)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    )


SubDict = Dict[str, Union[Data, "SubDict"]]
SubDictLike = Mapping[str, Union[DataLike, "SubDictLike"]]
MutableSubDict = MutableMapping[str, Union[Data, "MutableSubDict"]]


File = Dict[Optional[str], Union[StandaloneData, Data, SubDict]]
FileLike = Mapping[Optional[str], Union[StandaloneDataLike, DataLike, SubDictLike]]
