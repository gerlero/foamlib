"""Type aliases for OpenFOAM data structures."""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from multicollections import MultiDict

from ._files.types import Dimensioned, DimensionSet

type Tensor = float | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.float64]]
"""An OpenFOAM scalar, vector, symmetric tensor, or full tensor."""
type TensorLike = (
    Tensor
    | Sequence[float | np.floating | np.integer]
    | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.floating | np.integer]]
)
"""Any type that could be interpreted as a :type:`Tensor`."""

type Field = (
    float | np.ndarray[tuple[int] | tuple[int, Literal[3, 6, 9]], np.dtype[np.floating]]
)
"""An OpenFOAM field of scalars, vectors, symmetric tensors, or full tensors."""
type FieldLike = Field | TensorLike | Sequence[TensorLike]
"""Any type that could be interpreted as a :type:`Field`."""

type Dict = dict[str, Data | Dict]
"""An OpenFOAM dictionary."""
type DictLike = Mapping[str, DataLike | DictLike]
"""Any mapping that could be interpreted as a :type:`Dict`."""

type KeywordEntry = tuple[DataEntry, Data | Dict]
"""An OpenFOAM keyword entry (i.e., a key-value pair)."""
type KeywordEntryLike = tuple[DataEntryLike, DataLike | DictLike]
"""Any 2-tuple that could be interpreted as a :type:`KeywordEntry`."""

type List = list[DataEntry | KeywordEntry | Dict]
"""An OpenFOAM list."""
type ListLike = Sequence[DataEntryLike | KeywordEntryLike | DictLike]
"""Any sequence that could be interpreted as a :type:`List`."""

type DimensionSetLike = DimensionSet | Sequence[int | float]
"""Any type that could be interpreted as a :class:`foamlib.DimensionSet`."""

type DataEntry = str | int | float | bool | Dimensioned | DimensionSet | List | Field
"""
A single OpenFOAM value.
"""
type DataEntryLike = (
    DataEntry | np.integer | np.floating | DimensionSetLike | ListLike | FieldLike
)
"""Any type that could be interpreted as a :type:`DataEntry`."""

type Data = DataEntry | tuple[DataEntry, DataEntry, *tuple[DataEntry, ...]]
"""A single OpenFOAM value, or multiple values as a tuple."""
type DataLike = (
    DataEntryLike | tuple[DataEntryLike, DataEntryLike, *tuple[DataEntryLike, ...]]
)
"""Any type that could be interpreted as a :type:`Data`."""

type StandaloneDataEntry = (
    DataEntry
    | np.ndarray[tuple[int], np.dtype[np.int64 | np.int32 | np.float64]]
    | np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64 | np.float32]]
    | list[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]]]
)
"""A single OpenFOAM value that can appear at the top level of a file."""
type StandaloneDataEntryLike = (
    StandaloneDataEntry
    | DataEntryLike
    | Sequence[int]
    | Sequence[float]
    | Sequence[np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]]
    | Sequence[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.integer]]]
    | Sequence[Sequence[int]]
)
"""Any type that could be interpreted as a :type:`StandaloneDataEntry`."""

type StandaloneData = (
    StandaloneDataEntry
    | tuple[
        StandaloneDataEntry,
        StandaloneDataEntry,
        *tuple[StandaloneDataEntry, ...],
    ]
)
"""One or more OpenFOAM values that can appear at the top level of a file."""
type StandaloneDataLike = (
    StandaloneDataEntryLike
    | tuple[
        StandaloneDataEntryLike,
        StandaloneDataEntryLike,
        *tuple[StandaloneDataEntryLike, ...],
    ]
)
"""Any type that could be interpreted as a :type:`StandaloneData`."""

type SubDict = dict[str, Data | SubDict | None] | MultiDict[str, Data | SubDict | None]
"""An OpenFOAM dictionary nested in a file."""
type SubDictLike = Mapping[str, "DataLike | SubDictLike | None"]
"""Any mapping that could be interpreted as a :type:`SubDict`."""

type FileDict = (
    dict[str | None, StandaloneData | Data | SubDict | None]
    | MultiDict[str | None, StandaloneData | Data | SubDict | None]
)
"""An entire OpenFOAM file as a :class:`dict` or :class:`MultiDict`."""
type FileDictLike = Mapping[
    str | None, StandaloneDataLike | DataLike | SubDictLike | None
]
"""Any mapping that could be interpreted as a :type:`FileDict`."""
