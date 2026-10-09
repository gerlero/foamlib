import contextlib
import sys
from collections.abc import Buffer, Mapping
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


def _encoded_switch(value: bool, /) -> bool:
    if not isinstance(value, bool):
        msg = f"expected a bool, got {value!r}"
        raise TypeError(msg)
    return bool(value)


def _encoded_int(value: int, /) -> int:
    if not isinstance(value, int):
        msg = f"expected an int, got {value!r}"
        raise TypeError(msg)
    return int(value)


def _encoded_float(value: float, /) -> float:
    if not isinstance(value, (float, int)):
        msg = f"expected float or int, got {value!r}"
        raise TypeError(msg)
    return float(value)


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
        _dump(ret, fp, keywords=None)

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
            ret = _encoded_field(arr, fp, format_=format_)
        case _:
            msg = f"expected a Field, got {value!r}"
            raise TypeError(msg)

    if fp is not None:
        _dump(ret, fp, keywords=_common.FIELD_KEYWORDS, format_=format_)  # ty: ignore[invalid-argument-type]

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
        _dump(ret, fp, keywords=None)

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
        _dump(ret, fp, keywords=keywords, format_=format_)
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
        _dump(ret, fp, keywords=keywords, format_=format_)
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
        _dump(ret, fp, keywords=(), format_=format_)
    return ret


def _encoded_keyword_entry(value: KeywordEntryLike, /) -> KeywordEntry:
    match value:
        case DimensionSet():
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)
        case tuple((k, {} as d)):
            return _encoded_token(k), _encoded_dict(d, keywords=(), format_=None)  # ty: ignore[invalid-argument-type]
        case tuple((k, v)):
            return _encoded_token(k), _encoded_data_entry(  # ty: ignore[invalid-argument-type]
                v,  # ty: ignore[invalid-argument-type]
                keywords=None,
                format_=None,
            )
        case _:
            msg = f"expected a KeywordEntry (2-tuple), got {value!r}"
            raise TypeError(msg)


def _encoded_list(value: ListLike, /) -> List:
    match value:
        case np.ndarray(shape=(_, *_)):
            return _encoded_list(value.tolist())
        case tuple():
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)
        case [*_]:
            ret: List = []
            for v in value:
                match v:
                    case {}:
                        ret.append(_encoded_dict(v, keywords=(), format_=None))  # ty: ignore[invalid-argument-type]
                    case tuple():
                        ret.append(_encoded_keyword_entry(v))  # ty: ignore[invalid-argument-type]
                    case _:
                        ret.append(_encoded_data_entry(v, keywords=None, format_=None))
            return ret
        case _:
            msg = f"expected a List (sequence), got {value!r}"
            raise TypeError(msg)


def _encoded_data_entry(
    value: DataEntryLike,
    /,
    *,
    keywords: tuple[str, ...] | None,
    format_: Literal["ascii", "binary"] | None,
) -> DataEntry:
    if isinstance(value, (Dimensioned, DimensionSet)):
        return value
    if isinstance(value, str):
        ret = _encoded_token(value)
        match ret:
            case "no" | "false" | "off":
                msg = f"{ret!r} will be stored as False"
                warn(msg, stacklevel=2)
                return False
            case "yes" | "true" | "on":
                msg = f"{ret!r} will be stored as True"
                warn(msg, stacklevel=2)
                return True
            case _:
                return ret
    if isinstance(value, bool):
        return _encoded_switch(value)
    if keywords == _common.FIELD_KEYWORDS:
        with contextlib.suppress(TypeError):
            return _encoded_field(value, format_=format_)  # ty: ignore[invalid-argument-type]
    if keywords == ("dimensions",):
        with contextlib.suppress(TypeError):
            return _encoded_dimension_set(value)  # ty: ignore[invalid-argument-type]
    if isinstance(value, int):
        return _encoded_int(value)
    if isinstance(value, float):
        return _encoded_float(value)
    with contextlib.suppress(TypeError):
        return _encoded_list(value)  # ty: ignore[invalid-argument-type]
    msg = f"expected a DataEntry, got {value!r}"
    raise TypeError(msg)


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
        _dump(ret, fp, keywords=keywords, format_=format_)
    return ret


def _encoded_standalone_data_entry(
    value: StandaloneDataEntryLike,
    /,
    *,
    format_: Literal["ascii", "binary"] | None,
) -> StandaloneDataEntry:
    match value:
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="i")):
            if format_ != "binary" or value.dtype not in (np.int32, np.int64):
                return value.astype(int, copy=False)
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_,), dtype=np.dtype(kind="f")):
            return value.astype(np.float64, copy=False)
        case np.ndarray(shape=(_, 3), dtype=np.dtype(kind="f")):
            if format_ != "binary" or value.dtype not in (np.float64, np.float32):
                return value.astype(float, copy=False)
            return value  # ty: ignore[invalid-return-type]
        case np.ndarray(shape=(_, 3 | 4), dtype=np.dtype(kind="i")):
            return list(value.astype(int, copy=False))
        case np.ndarray() | Dimensioned() | DimensionSet() | tuple():
            pass
        case [*_]:
            try:
                arr = np.array(value)
            except (ValueError, TypeError):
                pass
            else:
                if arr.dtype in (int, float):
                    return _encoded_standalone_data_entry(arr, format_=format_)
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
        return _encoded_data_entry(value, keywords=(), format_=format_)  # ty: ignore[invalid-argument-type]
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
        _dump(ret, fp, keywords=(), format_=format_)
    return ret


class _PrefixedWriter[B: Buffer](Writer[B]):
    def __init__(self, writer: Writer, prefix: bytes, /) -> None:
        self._writer = writer
        self._prefix = prefix

    def write(self, b: B, /) -> int:
        if self._prefix and b:
            self._writer.write(self._prefix)
            self._prefix = b""
        return self._writer.write(b)


def _dump(
    data: FileDict
    | Data
    | StandaloneData
    | KeywordEntry
    | SubDict
    | Dict
    | np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]],
    fp: Writer[bytes],
    /,
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
    _tuple_is_keyword_entry: bool = False,
) -> None:
    match data, keywords, format_:
        case {"FoamFile": {"format": ("ascii" | "binary") as format_}}, (), None:  # ty: ignore[invalid-assignment]
            pass
    match data, keywords, format_:
        case {}, _, _:
            if keywords != ():
                fp.write(b"{")
            first = True
            for k, v in data.items():
                if not first:
                    fp.write(b" ")
                first = False
                if k is not None:
                    _dump(
                        (k, v),  # ty: ignore[invalid-argument-type]
                        fp,
                        keywords=keywords,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                else:
                    _dump(
                        v,  # ty: ignore[invalid-argument-type]
                        fp,
                        keywords=keywords,
                        format_=format_,
                    )
            if keywords != ():
                fp.write(b"}")

        case float(), _common.FIELD_KEYWORDS, _:
            fp.write(b"uniform ")
            _dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(3,) | (6,) | (9,)), _common.FIELD_KEYWORDS, _:
            fp.write(b"uniform ")
            _dump(data.tolist(), fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]
        case np.ndarray(shape=(_,)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<scalar> ")
            _dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(_, 3)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<vector> ")
            _dump(data, fp, keywords=None, format_=format_)
        case np.ndarray(shape=(_, 6)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<symmTensor> ")
            _dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(_, 9)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<tensor> ")
            _dump(data, fp, keywords=None, format_=format_)
        case np.ndarray(), _, "binary":
            _dump(len(data), fp, keywords=None, format_=None)
            fp.write(b"(")
            fp.write(data.tobytes())
            fp.write(b")")

        case np.ndarray(), (_, *_) | None, "ascii" | None:
            _dump(len(data), fp, keywords=None, format_=None)
            _dump(
                data.tolist(),  # ty: ignore[invalid-argument-type]
                fp,
                keywords=None,
                format_=format_,
            )
        case np.ndarray(), (), "ascii" | None:
            _dump(data.tolist(), fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]

        case DimensionSet(), _, _:
            try:
                name = _NAMED_DIMENSION_IDS[id(data)]
            except KeyError:
                fp.write(b"[")
                _dump(tuple(data), fp, keywords=None, format_=format_)
                fp.write(b"]")
            else:
                fp.write(b"[")
                _dump(name, fp, keywords=None, format_=format_)
                fp.write(b"]")
        case Dimensioned(name=None), _, _:
            _dump(data.dimensions, fp, keywords=None, format_=format_)
            fp.write(b" ")
            _dump(data.value, fp, keywords=None, format_=format_)
        case Dimensioned(name=str()), _, _:
            _dump(data.name, fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]
            fp.write(b" ")
            _dump(data.dimensions, fp, keywords=None, format_=format_)
            fp.write(b" ")
            _dump(data.value, fp, keywords=None, format_=format_)
        case (
            tuple((_, _)),
            _,
            _,
        ) if _tuple_is_keyword_entry and not isinstance(data, DimensionSet):
            assert len(data) == 2
            k, v = data
            if isinstance(k, str) and k[0] == "#":
                fp.write(b"\n")
            if k is not None:
                _dump(k, fp, keywords=keywords)
            _dump(
                v,
                _PrefixedWriter(fp, b" ") if k is not None else fp,
                keywords=(*keywords, k)  # ty: ignore[invalid-argument-type]
                if keywords is not None and k is not None
                else ()
                if k is None
                else None,
                format_=format_,
            )
            if isinstance(k, str) and k[0] == "#":
                fp.write(b"\n")
            elif (
                k is not None
                and not isinstance(v, Mapping)
                and not (isinstance(k, str) and k.startswith("$") and v is None)
            ):
                fp.write(b";")

        case tuple((_, _, *_)), _, _ if not isinstance(data, DimensionSet):
            for i, v in enumerate(data):
                if i:
                    fp.write(b" ")
                _dump(v, fp, keywords=keywords, format_=format_)
        case [*_], _, _:
            fp.write(b"(")
            for i, v in enumerate(data):
                if i:
                    fp.write(b" ")
                _dump(
                    v,  # ty: ignore[invalid-argument-type]
                    fp,
                    keywords=None,
                    format_=format_,
                    _tuple_is_keyword_entry=True,
                )
            fp.write(b")")

        case None, _, _:
            pass
        case True, _, _:
            fp.write(b"yes")

        case False, _, _:
            fp.write(b"no")

        case float() | int(), _, _:
            fp.write(str(data).encode())

        case str(), _, _:
            fp.write(data.encode())

        case _:
            assert_never(data)  # ty: ignore[type-assertion-failure]


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
