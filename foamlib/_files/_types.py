from __future__ import annotations

import sys
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping, Sequence
else:
    from typing import Iterator, Mapping, MutableMapping, Sequence

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
            self.value: Tensor = np.array(value, dtype=float)  # type: ignore [assignment]
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
            self.value + other.value,  # type: ignore [arg-type]
            self.dimensions + other.dimensions,
            f"{self.name}+{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __sub__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value - other.value,  # type: ignore [arg-type]
            self.dimensions - other.dimensions,
            f"{self.name}-{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __mul__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value * other.value,  # type: ignore [arg-type]
            self.dimensions * other.dimensions,
            f"{self.name}*{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __truediv__(self, other: Dimensioned | Tensor) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value / other.value,  # type: ignore [arg-type]
            self.dimensions / other.dimensions,
            f"{self.name}/{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __pow__(self, exponent: float) -> Dimensioned:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return Dimensioned(
            self.value**exponent,  # type: ignore [arg-type]
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


DataEntry = Union[
    str,
    int,
    float,
    bool,
    Dimensioned,
    DimensionSet,
    List[Union["DataEntry", Tuple["DataEntry", Union["DataEntry", "SubDict"]]]],
    Field,
]

DataEntryLike = Union[
    DataEntry,
    Sequence[
        Union[
            "DataEntryLike",
            Tuple["DataEntryLike", Union["DataEntryLike", "SubDictLike"]],
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


T = TypeVar("T", np.int64, np.int32)


class FaceList(Generic[T], Sequence["np.ndarray[tuple[int], np.dtype[T]]"]):
    indices: np.ndarray[tuple[int], np.dtype[T]]
    values: np.ndarray[tuple[int], np.dtype[T]]

    @overload
    def __init__(
        self,
        values: np.ndarray[tuple[int], np.dtype[T]] | Sequence[T | int],
        indices: np.ndarray[tuple[int], np.dtype[T]] | Sequence[T | int],
    ) -> None: ...

    @overload
    def __init__(
        self,
        values: np.ndarray[tuple[int, int], np.dtype[T]]
        | Sequence[np.ndarray[tuple[int], np.dtype[T]] | Sequence[T | int]],
    ) -> None: ...

    def __init__(
        self,
        values: np.ndarray[tuple[int], np.dtype[T]]
        | Sequence[T | int]
        | np.ndarray[tuple[int, int], np.dtype[T]]
        | Sequence[Sequence[T | int] | np.ndarray[tuple[int], np.dtype[T]]],
        indices: np.ndarray[tuple[int], np.dtype[T]] | Sequence[T | int] | None = None,
    ) -> None:
        if indices is None:
            dtype = values.dtype if isinstance(values, np.ndarray) else np.int64
            self.indices = np.empty(len(values) + 1, dtype=dtype)
            self.indices[0] = 0
            self.indices[1:] = np.cumsum([len(v) for v in values])  # type: ignore [arg-type]
            self.values = np.concatenate(values)  # type: ignore [arg-type, assignment]
        else:
            self.values = np.array(values)  # type: ignore [assignment]
            self.indices = np.array(indices, dtype=self.values.dtype)  # type: ignore [assignment]

    @overload
    def __getitem__(self, index: int) -> np.ndarray[tuple[int], np.dtype[T]]: ...

    @overload
    def __getitem__(
        self, index: slice
    ) -> Sequence[np.ndarray[tuple[int], np.dtype[T]]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> (
        np.ndarray[tuple[int], np.dtype[T]]
        | Sequence[np.ndarray[tuple[int], np.dtype[T]]]
    ):
        if isinstance(index, slice):
            msg = "Slicing FaceList is not yet implemented."
            raise NotImplementedError(msg)

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            msg = f"Index {index} out of range for FaceList of length {len(self)}"
            raise IndexError(msg)
        start = self.indices[index]
        end = self.indices[index + 1]
        return self.values[start:end]  # type: ignore [return-value]

    def __len__(self) -> int:
        return len(self.indices) - 1

    def __iter__(self) -> Iterator[np.ndarray[tuple[int], np.dtype[T]]]:
        for i in range(len(self)):
            yield self[i]

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, FaceList):
            return np.array_equal(self.values, other.values) and np.array_equal(
                self.indices, other.indices
            )

        if isinstance(other, Sequence):
            if len(self) != len(other):
                return False
            return all(np.array_equal(a, b) for a, b in zip(self, other))

        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.values}, {self.indices})"

    def tolist(self) -> list[list[int]]:
        return [self[i].tolist() for i in range(len(self))]


StandaloneData = Union[
    Data,
    "np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]",
    "np.ndarray[tuple[int], np.dtype[np.float64 | np.float32]]",
    FaceList[np.int64],
    FaceList[np.int32],
]

StandaloneDataLike = Union[
    DataLike,
    "np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]",
    "np.ndarray[tuple[int], np.dtype[np.float64 | np.float32]]",
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

File = Dict[Optional[str], Union[StandaloneData, Data, "SubDict"]]
FileLike = Mapping[Optional[str], Union[StandaloneDataLike, DataLike, "FileLike"]]
