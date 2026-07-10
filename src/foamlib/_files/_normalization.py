import contextlib
import sys
from collections.abc import Mapping
from typing import overload
from warnings import warn

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

import numpy as np

from .._files import _common
from ..typing import (
    Data,
    DataEntry,
    DataEntryLike,
    DataLike,
    Dict,
    DictLike,
    DimensionSetLike,
    Field,
    FieldLike,
    FileDict,
    FileDictLike,
    KeywordEntry,
    KeywordEntryLike,
    List,
    ListLike,
    StandaloneData,
    StandaloneDataEntry,
    StandaloneDataEntryLike,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
    Tensor,
    TensorLike,
)
from ._parsing import FoamFileDecodeError, parse
from ._util import add_to_mapping
from .types import Dimensioned, DimensionSet


def _normalized_str(value: str, /, *, keywords: tuple[str, ...] | None) -> str | bool:
    if not isinstance(value, str):
        msg = f"expected a string, got {value!r}"
        raise TypeError(msg)
    try:
        parsed = parse(
            value,
            target=StandaloneDataEntry if keywords == () else DataEntry,  # ty: ignore[invalid-argument-type]
        )
    except FoamFileDecodeError:
        msg = f"invalid string: {value!r}"
        raise ValueError(msg) from None
    match parsed:
        case "":
            msg = "empty string cannot be stored"
            raise ValueError(msg)
        case str():
            return parsed
        case bool():
            msg = f"{value!r} will be stored as {parsed!r}"
            warn(msg, stacklevel=2)
            return parsed
        case _:
            msg = f"{value!r} cannot be stored as a string (would be stored as {parsed!r})"
            raise ValueError(msg)


def _normalized_bool(value: bool, /) -> bool:  # noqa: FBT001
    if not isinstance(value, bool):
        msg = f"expected a bool, got {value!r}"
        raise TypeError(msg)
    return value


def _normalized_int(value: int, /) -> int:
    if not isinstance(value, int):
        msg = f"expected an int, got {value!r}"
        raise TypeError(msg)
    return value


def _normalized_float(value: float, /) -> float:
    if not isinstance(value, (float, int)):
        msg = f"expected float or int, got {value!r}"
        raise TypeError(msg)
    return float(value)


def _normalized_tensor(value: TensorLike, /) -> Tensor:
    match value:
        case float() | int():
            return float(value)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.dtype(kind="f" | "i")):
            return value.astype(float, copy=False)  # ty: ignore[no-matching-overload]
        case [*_] if len(value) in (3, 6, 9) and all(
            isinstance(v, (float, int)) for v in value
        ):
            return np.array(value, dtype=float)
        case _:
            msg = f"expected a Tensor, got {value!r}"
            raise TypeError(msg)


def _normalized_field(value: FieldLike, /, *, binary: bool | None = None) -> Field:
    match value:
        case float() | int():
            return float(value)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.dtype(kind="f" | "i")):
            return value.astype(float, copy=False)  # ty: ignore[no-matching-overload]
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.dtype(kind="f" | "i")
        ):
            if not binary or value.dtype not in (np.float64, np.float32):
                return value.astype(float, copy=False)  # ty: ignore[no-matching-overload]
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray():
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)
        case [*_]:
            try:
                arr = np.array(value, dtype=float)
            except (ValueError, TypeError):
                msg = f"expected a Field, got {value!r}"
                raise TypeError(msg) from None
            return _normalized_field(arr, binary=binary)
        case _:
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)


def _normalized_dimension_set(value: DimensionSetLike, /) -> DimensionSet:
    match value:
        case DimensionSet():
            return value
        case [*_] if len(value) <= 7 and all(
            isinstance(d, (int, float)) for d in value
        ):
            return DimensionSet(*value)
        case _:
            msg = f"expected a DimensionSet, got {value!r}"
            raise TypeError(msg)


def _normalized_dict(
    value: DictLike,
    /,
    *,
    binary: bool | None = None,
) -> Dict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    ret: Dict = {}
    for k, v in value.items():
        match k:
            case str():
                if k != _normalized_str(k, keywords=None):
                    msg = f"invalid keyword: {k!r}"
                    raise ValueError(msg)
                if k.startswith("#"):
                    msg = f"#-directive {k!r} not allowed here"
                    raise ValueError(msg)
                if k in ret:
                    warn(
                        f"Duplicate dictionary keyword found: {k!r}. Only the last entry will be stored.",
                        stacklevel=2,
                    )
                    del ret[k]
            case _:
                msg = f"invalid keyword: {k!r}"
                raise TypeError(msg)
        match v:
            case {}:
                ret[k] = _normalized_dict(v, binary=binary)
            case _:
                ret[k] = _normalized_data(v, keywords=None, binary=binary)
    return ret


def _normalized_subdict(
    value: SubDictLike,
    /,
    *,
    keywords: tuple[str, Unpack[tuple[str, ...]]],
    binary: bool | None = None,
) -> SubDict:
    assert keywords
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    ret: SubDict = {}
    for k, v in value.items():
        match k:
            case str():
                if k != _normalized_str(k, keywords=None):
                    msg = f"invalid keyword: {k!r}"
                    raise ValueError(msg)
                if k.startswith("#") and isinstance(v, Mapping):
                    msg = f"#-directive {k!r} cannot have a mapping as value; got value {v!r}"
                    raise TypeError(msg)
                if not k.startswith("#") and k in ret:
                    warn(
                        f"Duplicate subdictionary keyword found: {k!r}. Only the last entry will be stored.",
                        stacklevel=2,
                    )
                    del ret[k]
            case _:
                msg = f"invalid keyword: {k!r}"
                raise TypeError(msg)
        match v:
            case {}:
                ret[k] = _normalized_subdict(v, keywords=(*keywords, k), binary=binary)
            case None:
                ret = add_to_mapping(ret, k, None)
            case _:
                ret = add_to_mapping(
                    ret,
                    k,
                    _normalized_data(v, keywords=(*keywords, k), binary=binary),
                )
    return ret


def _normalized_file_dict(
    value: FileDictLike,
    /,
    *,
    binary: bool | None = None,
) -> FileDict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    if binary is None:
        match value:
            case {"FoamFile": {"format": "binary"}}:
                binary = True
            case {"FoamFile": {"format": "ascii"}}:
                binary = False
    ret: FileDict = {}
    for k, v in value.items():
        match k:
            case None:
                if None in ret:
                    msg = "duplicate None keyword found"
                    raise ValueError(msg)
            case str():
                if k != _normalized_str(k, keywords=None):
                    msg = f"invalid keyword: {k!r}"
                    raise ValueError(msg)
                if k.startswith("#") and isinstance(v, Mapping):
                    msg = f"#-directive {k!r} cannot have a mapping as value; got value {v!r}"
                    raise TypeError(msg)
                if not k.startswith("#") and k in ret:
                    warn(
                        f"Duplicate file keyword found: {k!r}. Only the last entry will be stored.",
                        stacklevel=2,
                    )
                    del ret[k]
            case _:
                msg = f"invalid keyword: {k!r}"
                raise TypeError(msg)
        match v:
            case {}:
                assert k is not None
                ret[k] = _normalized_subdict(v, keywords=(k,), binary=binary)  # ty: ignore[invalid-argument-type]
            case None:
                if k is None:
                    msg = "None keyword cannot have None value"
                    raise TypeError(msg)
                ret = add_to_mapping(ret, k, None)
            case _:
                if k is None:
                    ret[None] = _normalized_standalone_data(v, binary=binary)
                else:
                    ret = add_to_mapping(
                        ret,
                        k,
                        _normalized_data(v, keywords=(k,), binary=binary),  # ty: ignore[invalid-argument-type]
                    )
    return ret


def _normalized_keyword_entry(
    value: KeywordEntryLike, /, *, binary: bool | None = None
) -> KeywordEntry:
    match value:
        case DimensionSet():
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)
        case tuple((k, {} as d)):
            return _normalized_str(k, keywords=None), _normalized_dict(d, binary=binary)  # ty: ignore[invalid-argument-type]
        case tuple((k, v)):
            return _normalized_str(k, keywords=None), _normalized_data_entry(  # ty: ignore[invalid-argument-type]
                v,  # ty: ignore[invalid-argument-type]
                keywords=None,
                binary=binary,
            )
        case _:
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)


def _normalized_list(value: ListLike, /, *, binary: bool | None = None) -> List:
    match value:
        case np.ndarray(shape=(_, *_)):
            return _normalized_list(value.tolist(), binary=binary)  # ty: ignore[no-matching-overload]
        case tuple():
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)
        case [*_]:
            ret: List = []
            for v in value:
                match v:
                    case {}:
                        ret.append(_normalized_dict(v, binary=binary))  # ty: ignore[invalid-argument-type]
                    case tuple():
                        ret.append(_normalized_keyword_entry(v, binary=binary))  # ty: ignore[invalid-argument-type]
                    case _:
                        ret.append(
                            _normalized_data_entry(v, keywords=None, binary=binary)
                        )
            return ret
        case _:
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)


def _normalized_data_entry(
    value: DataEntryLike,
    /,
    *,
    keywords: tuple[str, ...] | None,
    binary: bool | None = None,
) -> DataEntry:
    if isinstance(value, (Dimensioned, DimensionSet)):
        return value
    if isinstance(value, str):
        return _normalized_str(value, keywords=keywords)
    if isinstance(value, bool):
        return _normalized_bool(value)
    if keywords == _common.FIELD_KEYWORDS:
        with contextlib.suppress(TypeError):
            return _normalized_field(value, binary=binary)  # ty: ignore[invalid-argument-type]
    if keywords == ("dimensions",):
        with contextlib.suppress(TypeError):
            return _normalized_dimension_set(value)  # ty: ignore[invalid-argument-type]
    if isinstance(value, int):
        return _normalized_int(value)
    if isinstance(value, float):
        return _normalized_float(value)
    with contextlib.suppress(TypeError):
        return _normalized_list(value, binary=binary)  # ty: ignore[invalid-argument-type]
    msg = f"expected a DataEntry, got {value!r}"
    raise TypeError(msg)


def _normalized_data(
    value: DataLike, /, *, keywords: tuple[str, ...] | None, binary: bool | None = None
) -> Data:
    match value:
        case DimensionSet():
            return value
        case tuple((_, _, *_)):
            return tuple(  # ty: ignore[invalid-return-type]
                _normalized_data_entry(v, keywords=keywords, binary=binary)  # ty: ignore[invalid-argument-type]
                for v in value
            )
        case _:
            return _normalized_data_entry(value, keywords=keywords, binary=binary)


def _normalized_standalone_data_entry(
    value: StandaloneDataEntryLike, /, *, binary: bool | None = None
) -> StandaloneDataEntry:
    match value:
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="i")):
            if not binary or value.dtype not in (np.int32, np.int64):
                return value.astype(int, copy=False)  # ty: ignore[no-matching-overload]
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="f")):
            return value.astype(np.float64, copy=False)  # ty: ignore[no-matching-overload]
        case np.ndarray(shape=(_, 3), dtype=np.dtype(kind="f")):
            if not binary or value.dtype not in (np.float64, np.float32):
                return value.astype(float, copy=False)  # ty: ignore[no-matching-overload]
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_, 3 | 4), dtype=np.dtype(kind="i")):
            return list(value.astype(int, copy=False))  # ty: ignore[no-matching-overload]
        case np.ndarray() | Dimensioned() | DimensionSet() | tuple():
            pass
        case [*_]:
            try:
                arr = np.array(value)
            except (ValueError, TypeError):
                pass
            else:
                if arr.dtype in (int, float):
                    return _normalized_standalone_data_entry(arr, binary=binary)
            ret = []
            for v in value:
                try:
                    e = np.asarray(v, dtype=int)
                except (ValueError, TypeError):
                    break
                if e.shape not in ((3,), (4,)):
                    break
                ret.append(e)
            else:
                return ret
    try:
        return _normalized_data_entry(value, keywords=(), binary=binary)  # ty: ignore[invalid-argument-type]
    except TypeError:
        msg = f"expected a StandaloneDataEntry, got {value!r}"
        raise TypeError(msg) from None


def _normalized_standalone_data(
    value: StandaloneDataLike, /, *, binary: bool | None = None
) -> StandaloneData:
    match value:
        case DimensionSet():
            return value
        case tuple((_, _, *_)):
            return tuple(  # ty: ignore[invalid-return-type]
                _normalized_standalone_data_entry(v, binary=binary)  # ty: ignore[invalid-argument-type]
                for v in value
            )
        case _:
            return _normalized_standalone_data_entry(value, binary=binary)


@overload
def normalized(
    value: str,
    /,
    *,
    target: type[str],
    keywords: tuple[str, ...] | None = ...,
) -> str | bool: ...


@overload
def normalized(
    value: DictLike,
    /,
    *,
    target: type[Dict],
    binary: bool | None = ...,
) -> Dict: ...


@overload
def normalized(
    value: SubDictLike,
    /,
    *,
    target: type[SubDict],
    keywords: tuple[str, ...],
    binary: bool | None = ...,
) -> SubDict: ...


@overload
def normalized(
    value: FileDictLike,
    /,
    *,
    target: type[FileDict],
    keywords: tuple[()] = ...,
    binary: bool | None = ...,
) -> FileDict: ...


@overload
def normalized(
    value: DataLike,
    /,
    *,
    target: type[Data],
    keywords: tuple[str, ...] | None = ...,
    binary: bool | None = ...,
) -> Data: ...


@overload
def normalized(
    value: StandaloneDataLike,
    /,
    *,
    target: type[StandaloneData] = ...,
    keywords: tuple[str, ...] | None = ...,
    binary: bool | None = ...,
) -> StandaloneData: ...


def normalized(
    value: str | DataLike | StandaloneDataLike | DictLike | SubDictLike | FileDictLike,
    /,
    *,
    target: type[
        str | Data | StandaloneData | Dict | SubDict | FileDict
    ] = StandaloneData,  # ty: ignore[invalid-parameter-default]
    keywords: tuple[str, ...] | None = (),
    binary: bool | None = None,
) -> object:
    if target is str:
        return _normalized_str(value, keywords=keywords)  # ty: ignore[invalid-argument-type]
    if target is Dict:
        return _normalized_dict(value, binary=binary)  # ty: ignore[invalid-argument-type]
    if target is SubDict:
        assert keywords
        return _normalized_subdict(value, keywords=keywords, binary=binary)
    if target is FileDict:
        assert keywords == ()
        return _normalized_file_dict(value, binary=binary)
    if target is Data:
        assert keywords != ()
        return _normalized_data(value, keywords=keywords, binary=binary)
    if target is StandaloneData:
        assert keywords == ()
        return _normalized_standalone_data(value, binary=binary)
    msg = f"unsupported target type: {target}"
    raise TypeError(msg)
