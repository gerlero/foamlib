"""Type aliases for OpenFOAM data structures."""

import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal, TypeAlias

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

import numpy as np
from multicollections import MultiDict

from ._files.types import Dimensioned, DimensionSet

Tensor: TypeAlias = float | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.float64]]
"""An OpenFOAM scalar, vector, symmetric tensor, or full tensor."""
TensorLike: TypeAlias = (
    Tensor
    | Real
    | Sequence[float | Real]
    | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.floating | np.integer]]
)
"""Any type that could be interpreted as a :type:`Tensor`."""

Field: TypeAlias = (
    float | np.ndarray[tuple[int] | tuple[int, Literal[3, 6, 9]], np.dtype[np.floating]]
)
"""An OpenFOAM field of scalars, vectors, symmetric tensors, or full tensors."""
FieldLike: TypeAlias = Field | Real | TensorLike | Sequence[TensorLike]
"""Any type that could be interpreted as a :type:`Field`."""

Dict: TypeAlias = dict[str, "Data | Dict"]
"""An OpenFOAM dictionary."""
DictLike: TypeAlias = Mapping[str, "DataLike | DictLike"]
"""Any mapping that could be interpreted as a :type:`Dict`."""

KeywordEntry: TypeAlias = tuple["DataEntry", "Data | Dict"]
"""An OpenFOAM keyword entry (i.e., a key-value pair)."""
KeywordEntryLike: TypeAlias = tuple["DataEntryLike", "Data | DictLike"]
"""Any 2-tuple that could be interpreted as a :type:`KeywordEntry`."""

List = list["DataEntry | KeywordEntry | Dict"]
"""An OpenFOAM list."""
ListLike: TypeAlias = Sequence["DataEntryLike | KeywordEntryLike | DictLike"]
"""Any sequence that could be interpreted as a :type:`List`."""

DimensionSetLike: TypeAlias = DimensionSet | Sequence[int | float]
"""Any type that could be interpreted as a :class:`foamlib.DimensionSet`."""

DataEntry: TypeAlias = (
    str | int | float | bool | Dimensioned | DimensionSet | List | Field
)
"""
A single OpenFOAM value.
"""
DataEntryLike: TypeAlias = (
    DataEntry | Integral | Real | DimensionSetLike | ListLike | FieldLike
)
"""Any type that could be interpreted as a :type:`DataEntry`."""

Data: TypeAlias = DataEntry | tuple[DataEntry, DataEntry, Unpack[tuple[DataEntry, ...]]]
"""A single OpenFOAM value, or multiple values as a tuple."""
DataLike: TypeAlias = (
    DataEntryLike
    | tuple[DataEntryLike, DataEntryLike, Unpack[tuple[DataEntryLike, ...]]]
)
"""Any type that could be interpreted as a :type:`Data`."""

StandaloneDataEntry: TypeAlias = (
    DataEntry
    | np.ndarray[tuple[int], np.dtype[np.int64 | np.int32 | np.float64]]
    | np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64 | np.float32]]
    | list[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]]]
)
"""A single OpenFOAM value that can appear at the top level of a file."""
StandaloneDataEntryLike: TypeAlias = (
    StandaloneDataEntry
    | DataEntryLike
    | Sequence[int]
    | Sequence[float]
    | Sequence[np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]]
    | Sequence[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.integer]]]
    | Sequence[Sequence[int]]
)
"""Any type that could be interpreted as a :type:`StandaloneDataEntry`."""

StandaloneData: TypeAlias = (
    StandaloneDataEntry
    | tuple[
        StandaloneDataEntry,
        StandaloneDataEntry,
        Unpack[tuple[StandaloneDataEntry, ...]],
    ]
)
"""One or more OpenFOAM values that can appear at the top level of a file."""
StandaloneDataLike: TypeAlias = (
    StandaloneDataEntryLike
    | tuple[
        StandaloneDataEntryLike,
        StandaloneDataEntryLike,
        Unpack[tuple[StandaloneDataEntryLike, ...]],
    ]
)
"""Any type that could be interpreted as a :type:`StandaloneData`."""

SubDict: TypeAlias = (
    dict[str, "Data | SubDict | None"] | MultiDict[str, "Data | SubDict | None"]
)
"""An OpenFOAM dictionary nested in a file."""
SubDictLike: TypeAlias = Mapping[str, "DataLike | SubDictLike | None"]
"""Any mapping that could be interpreted as a :type:`SubDict`."""

FileDict: TypeAlias = (
    dict[str | None, StandaloneData | Data | SubDict | None]
    | MultiDict[str | None, StandaloneData | Data | SubDict | None]
)
"""An entire OpenFOAM file as a :class:`dict` or :class:`MultiDict`."""
FileDictLike: TypeAlias = Mapping[
    str | None, StandaloneDataLike | DataLike | SubDictLike | None
]
"""Any mapping that could be interpreted as a :type:`FileDict`."""
