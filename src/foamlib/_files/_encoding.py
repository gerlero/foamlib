import contextlib
import sys
from collections.abc import Mapping
from typing import Literal, overload
from warnings import warn

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

if sys.version_info >= (3, 15):
    from typing import TypeForm
else:
    from typing_extensions import TypeForm

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
from .types import _NAMED_DIMENSION_IDS, Dimensioned, DimensionSet


def _write(fp: Writer[bytes] | None, data: bytes, /) -> None:
    if fp is not None:
        fp.write(data)


def _write_sequence(fp: Writer[bytes] | None, values: list, /) -> None:
    """Write the numeric nested sequences obtained from ndarray.tolist()."""
    _write(fp, b"(")
    for i, value in enumerate(values):
        if i:
            _write(fp, b" ")
        if isinstance(value, list):
            _write_sequence(fp, value)
        else:
            _write(fp, str(value).encode())
    _write(fp, b")")


def _write_array(
    fp: Writer[bytes] | None,
    value: np.ndarray,
    /,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> None:
    if fp is None:
        return
    if format_ == "binary":
        _write(fp, str(len(value)).encode())
        _write(fp, b"(")
        _write(fp, value.tobytes())
        _write(fp, b")")
    else:
        if keywords != ():
            _write(fp, str(len(value)).encode())
        _write_sequence(fp, value.tolist())


def _write_faces(
    fp: Writer[bytes] | None,
    faces: list[np.ndarray],
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> None:
    _write(fp, b"(")
    for i, face in enumerate(faces):
        if i:
            _write(fp, b" ")
        _write_array(fp, face, keywords=None, format_=format_)
    _write(fp, b")")


def _write_dimension_set(fp: Writer[bytes] | None, value: DimensionSet, /) -> None:
    if fp is None:
        return
    _write(fp, b"[")
    try:
        name = _NAMED_DIMENSION_IDS[id(value)]
    except KeyError:
        _write(fp, b" ".join(str(v).encode() for v in value))
    else:
        _write(fp, name.encode())
    _write(fp, b"]")


def _write_dimensioned(
    fp: Writer[bytes] | None,
    value: Dimensioned,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> None:
    if fp is None:
        return
    if value.name is not None:
        _write(fp, value.name.encode())
        _write(fp, b" ")
    _write_dimension_set(fp, value.dimensions)
    _write(fp, b" ")
    if isinstance(value.value, np.ndarray):
        _write_array(fp, value.value, keywords=None, format_=format_)
    else:
        _write(fp, str(value.value).encode())


def _write_keyword_start(
    fp: Writer[bytes] | None, key: str | None, value: object, /
) -> None:
    if fp is not None and key is not None:
        if key.startswith("#"):
            _write(fp, b"\n")
        _write(fp, key.encode())
        if value is not None:
            _write(fp, b" ")


def _write_keyword_end(
    fp: Writer[bytes] | None, key: str | None, value: object, /
) -> None:
    if fp is not None and key is not None:
        if key.startswith("#"):
            _write(fp, b"\n")
        elif not isinstance(value, Mapping) and not (
            key.startswith("$") and value is None
        ):
            _write(fp, b";")


def _last_indices(items: list[tuple], /, *, directives: bool) -> dict[str, int]:
    """Account for non-directive keywords overwritten by later entries."""
    return {
        key: i
        for i, (key, _) in enumerate(items)
        if isinstance(key, str) and (not directives or not key.startswith("#"))
    }


def _encoded_token(value: str, fp: Writer[bytes] | None = None, /) -> str:
    if not isinstance(value, str):
        msg = f"expected a string, got {value!r}"
        raise TypeError(msg)
    try:
        parsed = parse(value, target=str)
    except FoamFileDecodeError:
        msg = f"invalid token: {value!r}"
        raise ValueError(msg) from None
    _write(fp, parsed.encode())
    return parsed


def _encoded_switch(value: bool, fp: Writer[bytes] | None = None, /) -> bool:
    if not isinstance(value, bool):
        msg = f"expected a bool, got {value!r}"
        raise TypeError(msg)
    ret = bool(value)
    _write(fp, b"yes" if ret else b"no")
    return ret


def _encoded_int(value: int, fp: Writer[bytes] | None = None, /) -> int:
    if not isinstance(value, int):
        msg = f"expected an int, got {value!r}"
        raise TypeError(msg)
    ret = int(value)
    _write(fp, str(ret).encode())
    return ret


def _encoded_float(value: float, fp: Writer[bytes] | None = None, /) -> float:
    if not isinstance(value, (float, int)):
        msg = f"expected float or int, got {value!r}"
        raise TypeError(msg)
    ret = float(value)
    _write(fp, str(ret).encode())
    return ret


def _encoded_tensor(value: TensorLike, fp: Writer[bytes] | None = None, /) -> Tensor:
    match value:
        case float() | int():
            ret: Tensor = float(value)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.dtype(kind="f" | "i")):
            ret = value.astype(float, copy=False)
        case [*_] if len(value) in (3, 6, 9) and all(
            isinstance(v, (float, int)) for v in value
        ):
            ret = np.array(value, dtype=float)
        case _:
            msg = f"expected a Tensor, got {value!r}"
            raise TypeError(msg)
    if isinstance(ret, np.ndarray):
        _write_array(fp, ret, keywords=(), format_=None)
    else:
        _write(fp, str(ret).encode())
    return ret


def _encoded_field(
    value: FieldLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> Field:
    binary = format_ == "binary"
    match value:
        case float() | int():
            ret = float(value)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.dtype(kind="f" | "i")):
            ret = value.astype(float, copy=False)
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.dtype(kind="f" | "i")
        ):
            if not binary or value.dtype not in (np.float64, np.float32):
                ret = value.astype(float, copy=False)
            else:
                ret = value
        case np.ndarray():
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)
        case [*_]:
            try:
                arr = np.array(value, dtype=float)
            except (ValueError, TypeError):
                msg = f"expected a Field, got {value!r}"
                raise TypeError(msg) from None
            return _encoded_field(arr, fp, format_=format_)
        case _:
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)

    if isinstance(ret, np.ndarray):
        match ret.shape:
            case (3,) | (6,) | (9,):
                _write(fp, b"uniform ")
                if fp is not None:
                    _write_sequence(fp, ret.tolist())
            case (_, 3):
                _write(fp, b"nonuniform List<vector> ")
                _write_array(fp, ret, keywords=None, format_=format_)
            case (_, 6):
                _write(fp, b"nonuniform List<symmTensor> ")
                _write_array(fp, ret, keywords=None, format_=format_)
            case (_, 9):
                _write(fp, b"nonuniform List<tensor> ")
                _write_array(fp, ret, keywords=None, format_=format_)
            case (_,):
                _write(fp, b"nonuniform List<scalar> ")
                _write_array(fp, ret, keywords=None, format_=format_)
    else:
        _write(fp, b"uniform ")
        _write(fp, str(ret).encode())
    return ret  # ty: ignore[invalid-return-type]


def _encoded_dimension_set(
    value: DimensionSetLike, fp: Writer[bytes] | None = None, /
) -> DimensionSet:
    match value:
        case DimensionSet():
            ret = value
        case [*_] if len(value) <= 7 and all(
            isinstance(d, (int, float)) for d in value
        ):
            ret = DimensionSet(*value)
        case _:
            msg = f"expected a DimensionSet, got {value!r}"
            raise TypeError(msg)
    _write_dimension_set(fp, ret)
    return ret


def _encoded_plain_dict(
    value: DictLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    format_: Literal["ascii", "binary"] | None = None,
) -> Dict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    ret: Dict = {}
    items = list(value.items())
    last = _last_indices(items, directives=False)
    _write(fp, b"{")
    first = True
    for i, (k, v) in enumerate(items):
        match k:
            case str():
                if k != _encoded_token(k):
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

        item_fp = fp if last[k] == i else None
        if item_fp is not None:
            if not first:
                _write(item_fp, b" ")
            first = False
            _write_keyword_start(item_fp, k, v)
        match v:
            case {}:
                ret[k] = _encoded_plain_dict(v, item_fp, format_=format_)  # ty: ignore[invalid-argument-type]
            case _:
                ret[k] = _encoded_data(v, item_fp, keywords=None, format_=format_)
        _write_keyword_end(item_fp, k, v)
    _write(fp, b"}")
    return ret


def _encoded_dict(
    value: FileDictLike | SubDictLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    keywords: tuple[str, ...],
    format_: Literal["ascii", "binary"] | None,
) -> FileDict | SubDict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    is_file = not keywords
    if is_file and format_ is None:
        match value:
            case {"FoamFile": {"format": ("ascii" | "binary") as format_}}:  # ty: ignore[invalid-assignment]
                pass

    ret: FileDict | SubDict = {}
    items = list(value.items())
    last = _last_indices(items, directives=True)
    if not is_file:
        _write(fp, b"{")
    first = True
    for i, (k, v) in enumerate(items):
        match k:
            case None if is_file:
                if None in ret:
                    msg = "duplicate None keyword found"
                    raise ValueError(msg)
            case str():
                if k != _encoded_token(k):
                    msg = f"invalid keyword: {k!r}"
                    raise ValueError(msg)
                if k.startswith("#") and isinstance(v, Mapping):
                    msg = f"#-directive {k!r} cannot have a mapping as value; got value {v!r}"
                    raise TypeError(msg)
                if not k.startswith("#") and k in ret:
                    kind = "file" if is_file else "subdictionary"
                    warn(
                        f"Duplicate {kind} keyword found: {k!r}. Only the last entry will be stored.",
                        stacklevel=2,
                    )
                    del ret[k]
            case _:
                msg = f"invalid keyword: {k!r}"
                raise TypeError(msg)

        item_fp = fp if k is None or k.startswith("#") or last[k] == i else None
        if item_fp is not None:
            if not first:
                _write(item_fp, b" ")
            first = False
            _write_keyword_start(item_fp, k, v)

        match v:
            case {}:
                assert k is not None
                child = _encoded_dict(
                    v,  # ty: ignore[invalid-argument-type]
                    item_fp,
                    keywords=(*keywords, k),
                    format_=format_,
                )
                ret[k] = child  # ty: ignore[invalid-assignment]
            case None:
                if k is None:
                    msg = "None keyword cannot have None value"
                    raise TypeError(msg)
                ret = add_to_mapping(ret, k, None)  # ty: ignore[no-matching-overload]
            case _:
                if k is None:
                    ret[None] = _encoded_data(v, item_fp, keywords=(), format_=format_)
                else:
                    ret = add_to_mapping(
                        ret,
                        k,
                        _encoded_data(  # ty: ignore[no-matching-overload]
                            v,
                            item_fp,
                            keywords=(*keywords, k),
                            format_=format_,
                        ),
                    )
        _write_keyword_end(item_fp, k, v)
    if not is_file:
        _write(fp, b"}")
    return ret


def _encoded_keyword_entry(
    value: KeywordEntryLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> KeywordEntry:
    match value:
        case DimensionSet():
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)
        case tuple((k, {} as d)):
            k = _encoded_data_entry(k, fp, keywords=None, format_=format_)
            if fp is not None:
                fp.write(b" ")
            d = _encoded_plain_dict(d, fp, format_=format_)  # ty: ignore[invalid-argument-type]
            ret: KeywordEntry = (k, d)
        case tuple((k, v)):
            assert not isinstance(v, Mapping)
            k = _encoded_data_entry(k, fp, keywords=None, format_=format_)
            if fp is not None:
                fp.write(b" ")
            v = _encoded_data(v, fp, keywords=None, format_=format_)
            if fp is not None:
                fp.write(b";")
            ret: KeywordEntry = (k, v)
        case _:
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)
    return ret


def _encoded_list(
    value: ListLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> List:
    match value:
        case np.ndarray(shape=(_, *_)):
            return _encoded_list(value.tolist(), fp, format_=format_)
        case tuple():
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)
        case [*_]:
            ret: List = []
            _write(fp, b"(")
            for i, v in enumerate(value):
                if i:
                    _write(fp, b" ")
                match v:
                    case {}:
                        ret.append(_encoded_plain_dict(v, fp, format_=format_))  # ty: ignore[invalid-argument-type]
                    case tuple():
                        ret.append(_encoded_keyword_entry(v, fp, format_=format_))  # ty: ignore[invalid-argument-type]
                    case _:
                        ret.append(
                            _encoded_data_entry(v, fp, keywords=None, format_=format_)
                        )
            _write(fp, b")")
            return ret
        case _:
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)


def _encoded_data_entry(
    value: DataEntryLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> DataEntry:
    if isinstance(value, Dimensioned):
        _write_dimensioned(fp, value, format_=format_)
        return value
    if isinstance(value, DimensionSet):
        _write_dimension_set(fp, value)
        return value
    if isinstance(value, str):
        ret = _encoded_token(value)
        match ret:
            case "no" | "false" | "off":
                msg = f"{ret!r} will be stored as False"
                warn(msg, stacklevel=2)
                return _encoded_switch(False, fp)
            case "yes" | "true" | "on":
                msg = f"{ret!r} will be stored as True"
                warn(msg, stacklevel=2)
                return _encoded_switch(True, fp)
            case _:
                _write(fp, ret.encode())
                return ret
    if isinstance(value, bool):
        return _encoded_switch(value, fp)
    if keywords == _common.FIELD_KEYWORDS:
        with contextlib.suppress(TypeError):
            return _encoded_field(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
    if keywords == ("dimensions",):
        with contextlib.suppress(TypeError):
            return _encoded_dimension_set(value, fp)  # ty: ignore[invalid-argument-type]
    if isinstance(value, int):
        return _encoded_int(value, fp)
    if isinstance(value, float):
        return _encoded_float(value, fp)
    with contextlib.suppress(TypeError):
        return _encoded_list(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
    msg = f"expected a DataEntry, got {value!r}"
    raise TypeError(msg)


def _encoded_standalone_data_entry(
    value: StandaloneDataEntryLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> StandaloneDataEntry:
    binary = format_ == "binary"
    match value:
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="i")):
            if not binary or value.dtype not in (np.int32, np.int64):
                ret = value.astype(int, copy=False)
            else:
                ret = value
            _write_array(fp, ret, keywords=(), format_=format_)
            return ret  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="f")):
            ret = value.astype(np.float64, copy=False)
            _write_array(fp, ret, keywords=(), format_=format_)
            return ret
        case np.ndarray(shape=(_, 3), dtype=np.dtype(kind="f")):
            if not binary or value.dtype not in (np.float64, np.float32):
                ret = value.astype(float, copy=False)
            else:
                ret = value
            _write_array(fp, ret, keywords=(), format_=format_)
            return ret  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_, 3 | 4), dtype=np.dtype(kind="i")):
            ret = list(value.astype(int, copy=False))
            _write_faces(fp, ret, format_=format_)
            return ret
        case np.ndarray() | Dimensioned() | DimensionSet() | tuple():
            pass
        case [*_]:
            try:
                arr = np.array(value)
            except (ValueError, TypeError):
                pass
            else:
                if arr.dtype in (int, float):
                    return _encoded_standalone_data_entry(arr, fp, format_=format_)
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
                _write_faces(fp, ret, format_=format_)
                return ret
    try:
        return _encoded_data_entry(value, fp, keywords=(), format_=format_)  # ty: ignore[invalid-argument-type]
    except TypeError:
        msg = f"expected a StandaloneDataEntry, got {value!r}"
        raise TypeError(msg) from None


@overload
def _encoded_data(
    value: DataLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    keywords: tuple[str, *tuple[str, ...]] | None,
    format_: Literal["ascii", "binary"] | None,
) -> Data: ...


@overload
def _encoded_data(
    value: StandaloneDataLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    keywords: tuple[()],
    format_: Literal["ascii", "binary"] | None,
) -> StandaloneData: ...


def _encoded_data(
    value: DataLike | StandaloneDataLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> Data | StandaloneData:
    match value:
        case DimensionSet():
            _write_dimension_set(fp, value)
            return value
        case tuple((_, _, *_)):
            ret = []
            for i, v in enumerate(value):
                if i:
                    _write(fp, b" ")
                if keywords == ():
                    ret.append(_encoded_standalone_data_entry(v, fp, format_=format_))  # ty: ignore[invalid-argument-type]
                else:
                    ret.append(
                        _encoded_data_entry(v, fp, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
                    )
            return tuple(ret)  # ty: ignore[invalid-return-type]
        case _:
            if keywords == ():
                return _encoded_standalone_data_entry(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
            return _encoded_data_entry(value, fp, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]


@overload
def encoded(
    value: str,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: type[str],
) -> str: ...


@overload
def encoded(
    value: TensorLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[Tensor],
) -> Tensor: ...


@overload
def encoded(
    value: FileDictLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[FileDict],
) -> FileDict: ...


@overload
def encoded(
    value: SubDictLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[SubDict],
    keywords: tuple[str, *tuple[str, ...]],
    format_: Literal["ascii", "binary"] | None = ...,
) -> SubDict: ...


@overload
def encoded(
    value: DataLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[Data],
    keywords: tuple[str, ...] | None = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> Data: ...


@overload
def encoded(
    value: StandaloneDataLike,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[StandaloneData] = StandaloneData,
    format_: Literal["ascii", "binary"] | None = ...,
) -> StandaloneData: ...


def encoded(
    value: object,
    fp: Writer[bytes] | None = None,
    /,
    *,
    target: TypeForm[
        str | Tensor | FileDict | SubDict | Data | StandaloneData
    ] = StandaloneData,
    keywords: tuple[str, ...] | None = None,
    format_: Literal["ascii", "binary"] | None = None,
) -> object:
    if target is str:
        assert keywords is None and format_ is None
        return _encoded_token(value, fp)  # ty: ignore[invalid-argument-type]
    if target is Tensor:
        assert keywords is None and format_ is None
        return _encoded_tensor(value, fp)  # ty: ignore[invalid-argument-type]
    if target is FileDict:
        assert keywords is None and format_ is None
        return _encoded_dict(value, fp, keywords=(), format_=None)  # ty: ignore[invalid-argument-type]
    if target is SubDict:
        assert keywords is not None
        return _encoded_dict(value, fp, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
    if target is Data:
        assert keywords is not None
        return _encoded_data(value, fp, keywords=keywords, format_=format_)  # ty: ignore[no-matching-overload]
    if target is StandaloneData:
        assert keywords is None
        return _encoded_data(value, fp, keywords=(), format_=format_)  # ty: ignore[no-matching-overload]
    msg = f"unsupported target type: {target}"
    raise TypeError(msg)
