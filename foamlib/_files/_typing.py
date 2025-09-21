from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from multicollections import MultiDict

    from .types import Dimensioned, DimensionSet


Tensor: TypeAlias = "float | np.ndarray[tuple[int], np.dtype[np.float64]]"
TensorLike: TypeAlias = "Tensor | Sequence[float]"

Field: TypeAlias = "float | np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.float64 | np.float32]]"
FieldLike: TypeAlias = "Field | TensorLike | Sequence[TensorLike]"

KeywordEntry: TypeAlias = "tuple[DataEntry, DataEntry | Dict]"
KeywordEntryLike: TypeAlias = "tuple[DataEntryLike, DataEntryLike | DictLike]"

DataEntry: TypeAlias = "str | int | float | bool | Dimensioned | DimensionSet | list[DataEntry | KeywordEntry] | Field"
DataEntryLike: TypeAlias = (
    "DataEntry | Sequence[DataEntryLike | KeywordEntryLike] | FieldLike"
)

Data: TypeAlias = "DataEntry | tuple[DataEntry, ...]"
DataLike: TypeAlias = "DataEntryLike | tuple[DataEntryLike, ...]"

StandaloneData: TypeAlias = (
    "Data"
    "| np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]"
    "| np.ndarray[tuple[int, int], np.dtype[np.float64 | np.float32]]"
    "| list[np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]]"
    "| tuple[np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]], np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]]"
)
StandaloneDataLike: TypeAlias = (
    "StandaloneData"
    "| DataLike"
    "| Sequence[np.ndarray[tuple[int], np.dtype[np.int64 | np.int32]]]"
    "| Sequence[Sequence[int]]"
    "| tuple[Sequence[int], Sequence[int]]"
)

Dict: TypeAlias = "dict[str, DataEntry | Dict]"
DictLike: TypeAlias = "Mapping[str, DataEntryLike | DictLike]"

SubDict: TypeAlias = "dict[str, Data | SubDict] | MultiDict[str, Data | SubDict]"
SubDictLike: TypeAlias = DictLike

File: TypeAlias = "dict[str | None, StandaloneData | Data | SubDict] | MultiDict[str | None, StandaloneData | Data | SubDict]"
FileLike: TypeAlias = "Mapping[str | None, StandaloneDataLike | DataLike | SubDictLike]"
