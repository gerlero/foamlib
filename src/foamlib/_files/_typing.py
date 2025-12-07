from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import numpy as np
from multicollections import MultiDict

from .types import Dimensioned, DimensionSet

Tensor: TypeAlias = float | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.float64]]
TensorLike: TypeAlias = (
    Tensor
    | Sequence[float]
    | np.ndarray[tuple[Literal[3, 6, 9]], np.dtype[np.floating | np.integer]]
)

Field: TypeAlias = (
    float | np.ndarray[tuple[int] | tuple[int, Literal[3, 6, 9]], np.dtype[np.floating]]
)
FieldLike: TypeAlias = Field | TensorLike | Sequence[TensorLike]

Dict: TypeAlias = dict[str, "Data | Dict"]
DictLike: TypeAlias = Mapping[str, "DataLike | DictLike"]

KeywordEntry: TypeAlias = tuple["DataEntry", "Data | Dict"]
KeywordEntryLike: TypeAlias = tuple["DataEntryLike", "Data | DictLike"]

DataEntry: TypeAlias = (
    str
    | int
    | float
    | bool
    | Dimensioned
    | DimensionSet
    | list["DataEntry | KeywordEntry | Dict"]
    | Field
)
DataEntryLike: TypeAlias = (
    DataEntry | Sequence["DataEntryLike | KeywordEntryLike | DictLike"] | FieldLike
)

Data: TypeAlias = DataEntry | tuple[DataEntry, ...]
DataLike: TypeAlias = DataEntryLike | tuple[DataEntryLike, ...]

StandaloneDataEntry: TypeAlias = (
    DataEntry
    | np.ndarray[tuple[int], np.dtype[np.int64 | np.int32 | np.float64]]
    | np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64 | np.float32]]
    | list[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]]]
)
StandaloneDataEntryLike: TypeAlias = (
    StandaloneDataEntry
    | DataEntryLike
    | Sequence[int]
    | Sequence[float]
    | Sequence[np.ndarray[tuple[Literal[3]], np.dtype[np.floating | np.integer]]]
    | Sequence[np.ndarray[tuple[Literal[3, 4]], np.dtype[np.integer]]]
    | Sequence[Sequence[int]]
)

StandaloneData: TypeAlias = StandaloneDataEntry | tuple[StandaloneDataEntry, ...]
StandaloneDataLike: TypeAlias = (
    StandaloneDataEntryLike | tuple[StandaloneDataEntryLike, ...]
)

SubDict: TypeAlias = (
    dict[str, "Data | SubDict | None"] | MultiDict[str, "Data | SubDict | None"]
)
SubDictLike: TypeAlias = Mapping[str, "DataLike | SubDictLike | None"]

File: TypeAlias = (
    dict[str | None, StandaloneData | Data | SubDict | None]
    | MultiDict[str | None, StandaloneData | Data | SubDict | None]
)
FileLike: TypeAlias = Mapping[
    str | None, StandaloneDataLike | DataLike | SubDictLike | None
]
