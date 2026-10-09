import contextlib
import sys
from collections.abc import Mapping
from typing import Literal, assert_never, overload
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


def _encoded_token(value: str, /, fp: Writer[bytes] | None = None) -> str:
    if not isinstance(value, str):
        msg = f"expected a string, got {value!r}"
        raise TypeError(msg)
    try:
        parsed = parse(value, target=str)
    except FoamFileDecodeError:
        msg = f"invalid token: {value!r}"
        raise ValueError(msg) from None

    if fp is not None:
        fp.write(parsed.encode())

    return parsed


def _encoded_switch(value: bool, /, fp: Writer[bytes] | None = None) -> bool:
    if not isinstance(value, bool):
        msg = f"expected a bool, got {value!r}"
        raise TypeError(msg)
    ret = bool(value)
    if fp is not None:
        fp.write(b"yes" if ret else b"no")
    return ret


def _encoded_int(value: int, /, fp: Writer[bytes] | None = None) -> int:
    if not isinstance(value, int):
        msg = f"expected an int, got {value!r}"
        raise TypeError(msg)
    ret = int(value)
    if fp is not None:
        fp.write(str(ret).encode())
    return ret


def _encoded_float(value: float, /, fp: Writer[bytes] | None = None) -> float:
    if not isinstance(value, (float, int)):
        msg = f"expected float or int, got {value!r}"
        raise TypeError(msg)
    ret = float(value)
    if fp is not None:
        fp.write(str(ret).encode())
    return ret


def _encoded_tensor(value: TensorLike, /, fp: Writer[bytes] | None = None) -> Tensor:
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

    if fp is not None:
        if isinstance(ret, float):
            _encoded_float(ret, fp)
        else:
            _encoded_list(ret.tolist(), fp)  # ty: ignore[invalid-argument-type]

    return ret


def _encoded_field(
    value: FieldLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> Field:
    match value:
        case float() | int():
            ret: Field = float(value)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.dtype(kind="f" | "i")):
            ret = value.astype(float, copy=False)
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.dtype(kind="f" | "i")
        ):
            if format_ != "binary" or value.dtype not in (np.float64, np.float32):
                ret = value.astype(float, copy=False)
            else:
                ret = value  # ty: ignore[invalid-return-type]
        case np.ndarray():
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)
        case [*_]:
            try:
                arr = np.array(value, dtype=float)
            except (ValueError, TypeError):
                msg = f"expected a Field, got {value!r}"
                raise TypeError(msg) from None
            ret = _encoded_field(arr, format_=format_)
        case _:
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)

    if fp is not None:
        match ret:
            case float():
                fp.write(b"uniform ")
                _encoded_float(ret, fp)
            case np.ndarray(shape=(3,) | (6,) | (9,)):
                fp.write(b"uniform ")
                _encoded_list(ret.tolist(), fp)  # ty: ignore[invalid-argument-type]
            case np.ndarray(shape=(_,) | (_, 3) | (_, 6) | (_, 9)) as arr:
                label = {1: "scalar", 3: "vector", 6: "symmTensor", 9: "tensor"}[
                    1 if arr.ndim == 1 else arr.shape[1]
                ]
                fp.write(f"nonuniform List<{label}> ".encode())
                _encoded_list(arr, fp, format_=format_, _n=len(arr))
            case _:
                assert_never(ret)

    return ret


def _encoded_dimension_set(
    value: DimensionSetLike, /, fp: Writer[bytes] | None = None
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

    if fp is not None:
        try:
            name = _NAMED_DIMENSION_IDS[id(ret)]
        except KeyError:
            fp.write(b"[")
            for i, v in enumerate(tuple(ret)):
                if i:
                    fp.write(b" ")
                if isinstance(v, int):
                    _encoded_int(v, fp)
                else:
                    _encoded_float(v, fp)
            fp.write(b"]")
        else:
            fp.write(b"[")
            _encoded_token(name, fp)
            fp.write(b"]")

    return ret


def _encoded_dict(
    value: DictLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    keywords: tuple[str, ...],
    format_: Literal["ascii", "binary"] | None,
) -> Dict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    ret: Dict = {}
    for k, v in value.items():
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
        match v:
            case {}:
                ret[k] = _encoded_dict(
                    v,  # ty: ignore[invalid-argument-type]
                    keywords=keywords,
                    format_=format_,
                )
            case _:
                ret[k] = _encoded_data(v, keywords=None, format_=format_)
    if fp is not None:
        if keywords != ():
            fp.write(b"{")
        first = True
        for k, v in ret.items():
            if not first:
                fp.write(b" ")
            first = False
            _encoded_keyword_entry((k, v), fp, keywords=keywords, format_=format_)
        if keywords != ():
            fp.write(b"}")
    return ret


def _encoded_subdict(
    value: SubDictLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    keywords: tuple[str, *tuple[str, ...]],
    format_: Literal["ascii", "binary"] | None,
) -> SubDict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    ret: SubDict = {}
    for k, v in value.items():
        match k:
            case str():
                if k != _encoded_token(k):
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
                ret[k] = _encoded_subdict(
                    v,  # ty: ignore[invalid-argument-type]
                    keywords=(*keywords, k),
                    format_=format_,
                )
            case None:
                ret = add_to_mapping(ret, k, None)  # ty: ignore[no-matching-overload]
            case _:
                ret = add_to_mapping(  # ty: ignore[no-matching-overload]
                    ret,
                    k,
                    _encoded_data(v, keywords=(*keywords, k), format_=format_),
                )
    if fp is not None:
        fp.write(b"{")
        first = True
        for k, v in ret.items():
            if not first:
                fp.write(b" ")
            first = False
            assert isinstance(k, str)
            if v is not None and not isinstance(v, Mapping):
                if k.startswith("#"):
                    fp.write(b"\n")
                _encoded_token(k, fp)
                fp.write(b" ")
                _encoded_data(
                    v,  # ty: ignore[invalid-argument-type]
                    fp,
                    keywords=(*keywords, k),
                    format_=format_,
                )
                if k.startswith("#"):
                    fp.write(b"\n")
                else:
                    fp.write(b";")
            else:
                _encoded_keyword_entry(
                    (k, v),
                    fp,
                    keywords=(*keywords, k),
                    format_=format_,
                )
        fp.write(b"}")
    return ret


def _encoded_file_dict(
    value: FileDictLike, /, fp: Writer[bytes] | None = None
) -> FileDict:
    if not isinstance(value, Mapping):
        msg = f"expected a mapping, got {value!r}"
        raise TypeError(msg)
    match value:
        case {"FoamFile": {"format": "binary"}}:
            format_: Literal["ascii", "binary"] | None = "binary"
        case {"FoamFile": {"format": "ascii"}}:
            format_ = "ascii"
        case _:
            format_ = None
    ret: FileDict = {}
    for k, v in value.items():
        match k:
            case None:
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
                ret[k] = _encoded_subdict(
                    v,  # ty: ignore[invalid-argument-type]
                    keywords=(k,),
                    format_=format_,
                )
            case None:
                if k is None:
                    msg = "None keyword cannot have None value"
                    raise TypeError(msg)
                ret = add_to_mapping(ret, k, None)  # ty: ignore[no-matching-overload]
            case _:
                if k is None:
                    ret[None] = _encoded_standalone_data(v, format_=format_)
                else:
                    ret = add_to_mapping(  # ty: ignore[no-matching-overload]
                        ret,
                        k,
                        _encoded_data(v, keywords=(k,), format_=format_),  # ty: ignore[invalid-argument-type]
                    )
    if fp is not None:
        first = True
        for k, v in ret.items():
            if not first:
                fp.write(b" ")
            first = False
            if k is None:
                _encoded_standalone_data(v, fp, format_=format_)  # ty: ignore[invalid-argument-type]
            elif v is not None and not isinstance(v, Mapping):
                if k.startswith("#"):
                    fp.write(b"\n")
                _encoded_token(k, fp)
                fp.write(b" ")
                _encoded_data(v, fp, keywords=(k,), format_=format_)  # ty: ignore[invalid-argument-type]
                if k.startswith("#"):
                    fp.write(b"\n")
                else:
                    fp.write(b";")
            else:
                _encoded_keyword_entry(
                    (k, v),
                    fp,
                    keywords=(k,),
                    format_=format_,
                )
    return ret


def _encoded_keyword_entry(
    value: KeywordEntryLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    keywords: tuple[str, ...] = (),
    format_: Literal["ascii", "binary"] | None = None,
) -> KeywordEntry:
    match value:
        case DimensionSet():
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)
        case tuple((k, {} as v)):
            k, v = _encoded_token(k), _encoded_dict(v, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
        case tuple((k, v)):
            k, v = _encoded_token(k), _encoded_data_entry(  # ty: ignore[invalid-argument-type]
                v,  # ty: ignore[invalid-argument-type]
                keywords=None,
                format_=format_,
            )
        case _:
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)

    if fp is not None:
        if isinstance(k, str) and k.startswith("#"):
            fp.write(b"\n")
        _encoded_data_entry(k, fp, keywords=None, format_=format_)
        fp.write(b" ")
        if isinstance(v, Mapping):
            _encoded_dict(
                v,  # ty: ignore[invalid-argument-type]
                fp,
                keywords=keywords,
                format_=format_,
            )
        else:
            _encoded_data_entry(v, fp, keywords=keywords, format_=format_)
        if isinstance(k, str) and k.startswith("#"):
            fp.write(b"\n")
        elif not isinstance(v, Mapping) and not (
            isinstance(k, str) and k.startswith("$") and v is None
        ):
            fp.write(b";")

    return k, v  # ty: ignore[invalid-return-type]


def _encoded_list(
    value: ListLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    format_: Literal["ascii", "binary"] | None = None,
    _n: int | None = None,
) -> List:
    if isinstance(value, np.ndarray) and fp is not None and format_ == "binary":
        if _n is not None:
            fp.write(str(_n).encode())
        fp.write(b"(")
        fp.write(value.tobytes())
        fp.write(b")")
        return _encoded_list(value)

    match value:
        case np.ndarray(shape=(_,)):
            ret = _encoded_list(value.tolist())
        case np.ndarray(shape=(_, 3 | 4), dtype=np.dtype(kind="i")):
            ret = _encoded_list(list(value))
        case np.ndarray(shape=(_, *_)):
            ret = _encoded_list(value.tolist())
        case tuple():
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)
        case [*_]:
            ret = []
            for v in value:
                match v:
                    case Mapping():
                        ret.append(_encoded_dict(v, keywords=(), format_=None))  # ty: ignore[invalid-argument-type]
                    case np.ndarray():
                        ret.append(v)  # ty: ignore[invalid-argument-type]
                    case tuple():
                        ret.append(_encoded_keyword_entry(v))  # ty: ignore[invalid-argument-type]
                    case _:
                        ret.append(_encoded_data_entry(v, keywords=None, format_=None))
        case _:
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)

    if fp is not None:
        if _n is not None:
            fp.write(str(_n).encode())
        fp.write(b"(")
        for i, v in enumerate(ret):
            if i:
                fp.write(b" ")
            match v:
                case Mapping():
                    fp.write(b"{")
                    first = True
                    for k, entry in v.items():
                        if not first:
                            fp.write(b" ")
                        first = False
                        _encoded_keyword_entry(
                            (k, entry),
                            fp,
                            keywords=(),
                            format_=format_,
                        )
                    fp.write(b"}")
                case np.ndarray():
                    _encoded_list(v, fp, _n=len(v), format_=format_)
                case tuple():
                    _encoded_keyword_entry(
                        v, fp, keywords=(), format_=format_
                    )
                case _:
                    _encoded_data_entry(
                        v,  # ty: ignore[invalid-argument-type]
                        fp,
                        keywords=None,
                        format_=format_,
                    )
        fp.write(b")")

    return ret


def _encoded_data_entry(
    value: DataEntryLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> DataEntry:
    if isinstance(value, (Dimensioned, DimensionSet)):
        ret = value
    elif isinstance(value, str):
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
                if fp is not None:
                    fp.write(ret.encode())
                return ret
    elif isinstance(value, bool):
        return _encoded_switch(value, fp)
    elif keywords == _common.FIELD_KEYWORDS:
        with contextlib.suppress(TypeError):
            return _encoded_field(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
        if isinstance(value, int):
            return _encoded_int(value, fp)
        if isinstance(value, float):
            return _encoded_float(value, fp)
        with contextlib.suppress(TypeError):
            return _encoded_list(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
        msg = f"expected a DataEntry, got {value!r}"
        raise TypeError(msg)
    elif keywords == ("dimensions",):
        with contextlib.suppress(TypeError):
            return _encoded_dimension_set(value, fp)
        if isinstance(value, int):
            return _encoded_int(value, fp)
        if isinstance(value, float):
            return _encoded_float(value, fp)
        with contextlib.suppress(TypeError):
            return _encoded_list(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
        msg = f"expected a DataEntry, got {value!r}"
        raise TypeError(msg)
    elif isinstance(value, int):
        return _encoded_int(value, fp)
    elif isinstance(value, float):
        return _encoded_float(value, fp)
    else:
        with contextlib.suppress(TypeError):
            return _encoded_list(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
        msg = f"expected a DataEntry, got {value!r}"
        raise TypeError(msg)

    if fp is not None:
        if isinstance(ret, DimensionSet):
            _encoded_dimension_set(ret, fp)
        else:
            assert isinstance(ret, Dimensioned)
            if ret.name is not None:
                _encoded_token(ret.name, fp)
                fp.write(b" ")
            _encoded_dimension_set(ret.dimensions, fp)
            fp.write(b" ")
            _encoded_tensor(ret.value, fp)
    return ret


def _encoded_data(
    value: DataLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> Data:
    match value:
        case DimensionSet():
            ret: Data = value
        case tuple((_, _, *_)):
            ret = tuple(  # ty: ignore[invalid-assignment]
                _encoded_data_entry(v, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
                for v in value
            )
        case _:
            ret = _encoded_data_entry(value, keywords=keywords, format_=format_)
    if fp is not None:
        if isinstance(ret, tuple) and not isinstance(ret, DimensionSet):
            first = True
            for v in ret:
                if not first:
                    fp.write(b" ")
                first = False
                _encoded_data_entry(v, fp, keywords=keywords, format_=format_)
        else:
            _encoded_data_entry(ret, fp, keywords=keywords, format_=format_)
    return ret


def _encoded_standalone_data_entry(
    value: StandaloneDataEntryLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> StandaloneDataEntry:
    match value:
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="i")):
            if format_ != "binary" or value.dtype not in (np.int32, np.int64):
                value = value.astype(int, copy=False)
            if fp is not None:
                if format_ == "binary":
                    fp.write(str(len(value)).encode())
                    fp.write(b"(")
                    fp.write(value.tobytes())
                    fp.write(b")")
                else:
                    _encoded_list(value.tolist(), fp, format_=format_)
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="f")):
            value = value.astype(np.float64, copy=False)
            if fp is not None:
                if format_ == "binary":
                    fp.write(str(len(value)).encode())
                    fp.write(b"(")
                    fp.write(value.tobytes())
                    fp.write(b")")
                else:
                    _encoded_list(value.tolist(), fp, format_=format_)
            return value
        case np.ndarray(shape=(_, 3), dtype=np.dtype(kind="f")):
            if format_ != "binary" or value.dtype not in (np.float64, np.float32):
                value = value.astype(float, copy=False)
            if fp is not None:
                if format_ == "binary":
                    fp.write(str(len(value)).encode())
                    fp.write(b"(")
                    fp.write(value.tobytes())
                    fp.write(b")")
                else:
                    _encoded_list(value.tolist(), fp, format_=format_)
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_, 3 | 4), dtype=np.dtype(kind="i")):
            value = list(value.astype(int, copy=False))
            if fp is not None:
                _encoded_list(
                    [v.tolist() for v in value], fp, format_=format_
                )
            return value
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
                if fp is not None:
                    _encoded_list([e.tolist() for e in ret], fp, format_=format_)
                return ret
    try:
        return _encoded_data_entry(value, fp, keywords=(), format_=format_)  # ty: ignore[invalid-argument-type]
    except TypeError:
        msg = f"expected a StandaloneDataEntry, got {value!r}"
        raise TypeError(msg) from None


def _encoded_standalone_data(
    value: StandaloneDataLike,
    /,
    fp: Writer[bytes] | None = None,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> StandaloneData:
    match value:
        case DimensionSet():
            ret: StandaloneData = value
        case tuple((_, _, *_)):
            ret = tuple(  # ty: ignore[invalid-assignment]
                _encoded_standalone_data_entry(v, format_=format_)  # ty: ignore[invalid-argument-type]
                for v in value
            )
        case _:
            ret = _encoded_standalone_data_entry(value, format_=format_)
    if fp is not None:
        if isinstance(ret, tuple) and not isinstance(ret, DimensionSet):
            first = True
            for v in ret:
                if not first:
                    fp.write(b" ")
                first = False
                _encoded_standalone_data_entry(v, fp, format_=format_)
        else:
            _encoded_standalone_data_entry(ret, fp, format_=format_)
    return ret


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
        return _encoded_file_dict(value, fp)  # ty: ignore[invalid-argument-type]
    if target is SubDict:
        assert keywords is not None
        return _encoded_subdict(value, fp, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
    if target is Data:
        assert keywords is not None
        return _encoded_data(value, fp, keywords=keywords, format_=format_)  # ty: ignore[invalid-argument-type]
    if target is StandaloneData:
        assert keywords is None
        return _encoded_standalone_data(value, fp, format_=format_)  # ty: ignore[invalid-argument-type]
    raise TypeError(f"unsupported target type: {target}")
