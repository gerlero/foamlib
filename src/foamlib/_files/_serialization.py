import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal, overload
from warnings import warn

if sys.version_info >= (3, 11):
    from typing import Never, Unpack, assert_never, assert_type
else:
    from typing_extensions import Never, Unpack, assert_never, assert_type

import numpy as np

from .._files import _common
from ..typing import (
    Data,
    DataLike,
    Dict,
    DictLike,
    FileDict,
    FileDictLike,
    KeywordEntry,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
)
from ._parsing import FoamFileDecodeError, parse
from ._util import add_to_mapping
from .types import Dimensioned, DimensionSet


@overload
def normalized(
    data: FileDictLike,
    *,
    keywords: tuple[()] = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> FileDict: ...


@overload
def normalized(
    data: DataLike,
    /,
    *,
    keywords: tuple[str, Unpack[tuple[str, ...]]] | None = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> Data: ...


@overload
def normalized(
    data: StandaloneDataLike,
    /,
    *,
    keywords: tuple[()] = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> StandaloneData: ...


@overload
def normalized(
    data: SubDictLike,
    /,
    *,
    keywords: tuple[str, Unpack[tuple[str, ...]]] = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> SubDict: ...


@overload
def normalized(
    data: DictLike,
    /,
    *,
    keywords: None = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> Data: ...


@overload
def normalized(
    data: None,
    /,
    *,
    keywords: tuple[str, Unpack[tuple[str, ...]]] = ...,
    format_: Literal["ascii", "binary"] | None = ...,
) -> None: ...


def normalized(
    data: FileDictLike | DataLike | StandaloneDataLike | SubDictLike | DictLike | None,
    /,
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
) -> FileDict | Data | StandaloneData | SubDict | Dict | None:
    match data, keywords, format_:
        case {"FoamFile": {"format": ("ascii" | "binary") as format_}}, (), None:
            pass

    match data, keywords:
        # File
        case {}, ():
            ret: FileDict = {}
            seen_none = False
            for k, v in data.items():  # ty: ignore[possibly-missing-attribute]
                normalized_v = normalized(
                    v,
                    keywords=(k,) if k is not None else (),  # ty: ignore[not-iterable]
                    format_=format_,
                )  # ty: ignore[no-matching-overload]

                match k:
                    case None:
                        if seen_none:
                            msg = "duplicate None keyword found"
                            raise ValueError(msg)
                        seen_none = True
                        ret[None] = normalized_v

                    case str():
                        if k != normalized(k):
                            msg = f"invalid keyword: {k!r}"
                            raise ValueError(msg)

                        if k.startswith("#"):
                            if isinstance(normalized_v, Mapping):
                                msg = f"#-directive {k!r} cannot have a mapping as value; got value {normalized_v!r}"
                                raise TypeError(msg)
                            ret = add_to_mapping(ret, k, normalized_v)  # ty: ignore[invalid-assignment]
                        else:
                            if k in ret:
                                msg = f"duplicate keyword found: {k!r}"
                                raise ValueError(msg)
                            ret[k] = normalized_v

                    case _:
                        msg = f"keyword must be a string or None; got {k!r}"
                        raise TypeError(msg)

            return ret  # ty: ignore[invalid-return-type]

        # Sub-dictionary
        case {}, (_, *_):
            ret: SubDict = {}
            for k, v in data.items():  # ty: ignore[possibly-missing-attribute]
                normalized_v = normalized(v, keywords=(*keywords, k), format_=format_)  # ty: ignore[no-matching-overload, not-iterable]

                match k:
                    case None:
                        msg = "None key is only allowed in top-level File dicts"
                        raise TypeError(msg)

                    case str():
                        if k != normalized(k):
                            msg = f"invalid keyword: {k!r}"
                            raise ValueError(msg)

                        if k.startswith("#"):
                            if isinstance(normalized_v, Mapping):
                                msg = f"#-directive {k!r} cannot have a mapping as value; got value {normalized_v!r}"
                                raise TypeError(msg)
                            ret = add_to_mapping(ret, k, normalized_v)  # ty: ignore[invalid-assignment]
                        else:
                            if k in ret:
                                msg = f"duplicate keyword found: {k!r}"
                                raise ValueError(msg)
                            ret[k] = normalized_v

                    case _:
                        msg = f"keyword must be a string; got {k!r}"
                        raise TypeError(msg)

            return ret

        # Other dictionary
        case {}, None:
            ret: Dict = {}
            for k, v in data.items():  # ty: ignore[possibly-missing-attribute]
                normalized_v = normalized(v, keywords=None, format_=format_)  # ty: ignore[no-matching-overload]

                match k:
                    case None:
                        msg = "None keyword is only allowed in top-level File dicts"
                        raise TypeError(msg)

                    case str():
                        if k != normalized(k):
                            msg = f"invalid keyword: {k!r}"
                            raise ValueError(msg)

                        if k.startswith("#"):
                            msg = f"#-directive {k!r} not allowed here"
                            raise ValueError(msg)

                        if k in ret:
                            msg = f"duplicate keyword found: {k!r}"
                            raise ValueError(msg)
                        ret[k] = normalized_v

                    case _:
                        msg = f"keyword must be a string; got {k!r}"
                        raise TypeError(msg)

            return ret

        # Numeric standalone data (n integers)
        case np.ndarray(shape=(_,)), () if np.issubdtype(data.dtype, np.integer):  # ty: ignore[possibly-missing-attribute]
            if format_ == "binary":
                if data.dtype != np.int32:  # ty: ignore[possibly-missing-attribute]
                    msg = f"only int32 data type is supported for this kind of binary data, got {data.dtype}"  # ty: ignore[possibly-missing-attribute]
                    raise ValueError(msg)
                return data  # ty: ignore[invalid-return-type]
            return data.astype(int, copy=False)  # ty: ignore[possibly-missing-attribute]

        # Numeric standalone data (n floats)
        case np.ndarray(shape=(_,)), () if np.issubdtype(data.dtype, np.floating):  # ty: ignore[possibly-missing-attribute]
            if format_ == "binary":
                if data.dtype != np.float64:  # ty: ignore[possibly-missing-attribute]
                    msg = f"only float64 data type is supported for this kind of binary data, got {data.dtype}"  # ty: ignore[possibly-missing-attribute]
                    raise ValueError(msg)
                return data  # ty: ignore[invalid-return-type]
            return data.astype(float, copy=False)  # ty: ignore[possibly-missing-attribute]

        # Numeric standalone data (n x 3 floats)
        case np.ndarray(shape=(_,) | (_, 3)), () if np.issubdtype(
            data.dtype,  # ty: ignore[possibly-missing-attribute]
            np.floating,
        ) or np.issubdtype(data.dtype, np.integer):  # ty: ignore[possibly-missing-attribute]
            if format_ == "binary":
                if data.dtype not in (np.float64, np.float32):  # ty: ignore[possibly-missing-attribute]
                    msg = f"only float64 or float32 data types are supported for this kind of binary data, got {data.dtype}"  # ty: ignore[possibly-missing-attribute]
                    raise ValueError(msg)
                return data  # ty: ignore[invalid-return-type]
            return data.astype(float, copy=False)  # ty: ignore[possibly-missing-attribute]

        # ASCII faces-like list
        case [*_], () if not isinstance(data, tuple) and all(
            isinstance(e, np.ndarray)
            and e.shape in ((3,), (4,))
            and e.dtype == np.int32
            for e in data  # ty: ignore[not-iterable]
        ):
            if not isinstance(data, list):
                data = list(data)  # ty: ignore[invalid-argument-type]
            return data  # ty: ignore[invalid-return-type]

        # Other sequence convertible to ASCII faces-like list
        case [*_], () if not isinstance(data, tuple) and all(
            (
                (
                    isinstance(e, np.ndarray)
                    and e.shape in ((3,), (4,))
                    and np.issubdtype(e.dtype, np.integer)
                )
                or (
                    isinstance(e, Sequence)
                    and len(e) in (3, 4)
                    and all(isinstance(n, Integral) for n in e)
                )
            )
            for e in data  # ty: ignore[not-iterable]
        ):
            return [np.asarray(e) for e in data]  # ty: ignore[not-iterable]

        # Other possible numeric standalone data (n integers or floats)
        case [Real(), *rest], () if not isinstance(data, tuple) and all(
            isinstance(r, Real) for r in rest
        ):
            return normalized(np.asarray(data), keywords=keywords, format_=format_)

        # Other possible numeric standalone data (n x 3 floats)
        case [_, *_], () if not isinstance(data, tuple) and all(
            (
                isinstance(r, Sequence)
                and not isinstance(r, tuple)
                and len(r) == 3
                and all(isinstance(x, Real) for x in r)
            )
            or (
                isinstance(r, np.ndarray)
                and r.shape == (3,)
                and (
                    np.issubdtype(r.dtype, np.floating)
                    or np.issubdtype(r.dtype, np.integer)
                )
            )
            for r in data  # ty: ignore[not-iterable]
        ):
            return normalized(np.asarray(data), keywords=keywords, format_=format_)

        # Uniform field (scalar)
        case Real(), _common.FIELD_KEYWORDS:
            return float(data)  # ty: ignore[invalid-argument-type]

        # Uniform field (non-scalar)
        case np.ndarray(shape=(3,) | (6,) | (9,)), _common.FIELD_KEYWORDS if (
            np.issubdtype(data.dtype, np.floating)  # ty: ignore[possibly-missing-attribute]
            or np.issubdtype(data.dtype, np.integer)  # ty: ignore[possibly-missing-attribute]
        ):
            return data.astype(float, copy=False)  # ty: ignore[possibly-missing-attribute]

        # Non-uniform field
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9)
        ), _common.FIELD_KEYWORDS if np.issubdtype(
            data.dtype,  # ty: ignore[possibly-missing-attribute]
            np.floating,
        ) or np.issubdtype(data.dtype, np.integer):  # ty: ignore[possibly-missing-attribute]
            if format_ == "binary":
                if np.issubdtype(data.dtype, np.integer):  # ty: ignore[possibly-missing-attribute]
                    msg = "binary fields cannot have an integer data type"
                    raise ValueError(msg)
                return data  # ty: ignore[invalid-return-type]
            return data.astype(float, copy=False)  # ty: ignore[possibly-missing-attribute]

        # Other possible uniform field
        case [Real(), Real(), Real()] | [
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
        ] | [
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
        ], _common.FIELD_KEYWORDS:
            return normalized(np.array(data), keywords=keywords, format_=format_)

        # Other possible non-uniform scalar or empty field
        case [*_], _common.FIELD_KEYWORDS if not isinstance(data, tuple) and all(
            isinstance(d, Real)
            for d in data  # ty: ignore[not-iterable]
        ):
            return normalized(
                np.array(data, dtype=float), keywords=keywords, format_=format_
            )

        # Other possible non-uniform non-scalar field
        case [
            [Real(), Real(), Real()]
            | [Real(), Real(), Real(), Real(), Real(), Real()]
            | [
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
            ] as first,
            *rest,
        ], _common.FIELD_KEYWORDS if not isinstance(data, tuple) and all(
            isinstance(r, Sequence)
            and not isinstance(r, tuple)
            and (all(isinstance(x, Real) for x in r) and (len(r) == len(first)))
            for r in rest
        ):
            return normalized(
                np.array(data, dtype=float), keywords=keywords, format_=format_
            )

        # Dimension set from list of numbers
        case [*_], ("dimensions",) if len(data) <= 7 and all(  # ty: ignore[invalid-argument-type]
            isinstance(d, Real)
            for d in data  # ty: ignore[not-iterable]
        ):
            return DimensionSet(*data)  # ty: ignore[invalid-argument-type]

        # List
        case [*_], _ if not isinstance(data, tuple):
            return [normalized(d, keywords=None, format_=format_) for d in data]  # ty: ignore[no-matching-overload,not-iterable]

        # Other Numpy array (treated as list)
        case np.ndarray(), _:
            return normalized(data.tolist(), keywords=keywords, format_=format_)  # ty: ignore[possibly-missing-attribute]

        # Multiple data entries or keyword entry (tuple)
        case (
            tuple((_, _, *_)),
            _,
        ) if not isinstance(data, DimensionSet):
            ret = tuple(normalized(d, keywords=keywords, format_=format_) for d in data)  # ty: ignore[no-matching-overload,not-iterable]
            if any(isinstance(d, tuple) for d in ret):
                msg = f"nested tuples not supported: {data!r}"
                raise ValueError(msg)
            return ret  # ty: ignore[invalid-return-type]

        # One-element tuple (unsupported)
        case tuple((_,)), _:
            msg = f"found unsupported one-element tuple: {data!r}"
            raise ValueError(msg)

        # Empty tuple (unsupported)
        case tuple(()), _:
            msg = "found unsupported empty tuple"
            raise ValueError(msg)

        # Top-level string
        case str(), ():
            try:
                match parsed := parse(data, target=StandaloneData):  # ty: ignore[invalid-argument-type]
                    case str():
                        if not parsed:
                            msg = "found unsupported empty string"
                            raise ValueError(msg)
                        return parsed  # ty: ignore[invalid-return-type]
                    case bool():
                        msg = f"{data!r} will be stored as {parsed!r}"
                        warn(msg, stacklevel=2)
                        return parsed
                    case tuple((str() | bool(), str() | bool(), *rest)) if all(
                        isinstance(p, (str, bool)) for p in rest
                    ):
                        msg = f"{data!r} will be stored as {parsed!r}"
                        warn(msg, stacklevel=2)
                        return parsed
                    case _:
                        msg = f"{data!r} cannot be stored as string (would be stored as {parsed!r})"
                        raise ValueError(msg)
            except FoamFileDecodeError:
                msg = f"invalid string: {data!r}"
                raise ValueError(msg) from None

        # String
        case str(), (_, *_) | None:
            try:
                match parsed := parse(data, target=Data):  # ty: ignore[invalid-argument-type]
                    case str():
                        if not parsed:
                            msg = "found unsupported empty string"
                            raise ValueError(msg)
                        return parsed  # ty: ignore[invalid-return-type]
                    case bool():
                        msg = f"{data!r} will be stored as {parsed!r}"
                        warn(msg, stacklevel=2)
                        return parsed
                    case tuple((str() | bool(), str() | bool(), *rest)) if all(
                        isinstance(p, (str, bool)) for p in rest
                    ):
                        msg = f"{data!r} will be stored as {parsed!r}"
                        warn(msg, stacklevel=2)
                        return parsed
                    case _:
                        msg = f"{data!r} cannot be stored as string (would be stored as {parsed!r})"
                        raise ValueError(msg)
            except FoamFileDecodeError:
                msg = f"invalid string: {data!r}"
                raise ValueError(msg) from None

        # None
        case None, (*_, _):
            return None

        # Boolean
        case True | False, _:
            return data  # ty: ignore[invalid-return-type]

        # Integer
        case Integral(), _:
            return int(data)  # ty: ignore[invalid-argument-type]

        # Float
        case Real(), _:
            return float(data)  # ty: ignore[invalid-argument-type]

        # Custom types
        case DimensionSet() | Dimensioned(), _:
            return data  # ty: ignore[invalid-return-type]

        # Unsupported type
        case _:
            assert_type(data, Never)  # ty: ignore[type-assertion-failure]
            msg = f"Unsupported data type: {data!r} ({type(data)})"
            raise TypeError(msg)


def dumps(
    data: FileDict | Data | StandaloneData | KeywordEntry | SubDict,
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
    _tuple_is_keyword_entry: bool = False,
) -> bytes:
    match data, keywords, format_:
        case {"FoamFile": {"format": ("ascii" | "binary") as format_}}, (), None:
            pass

    match data, keywords, format_:
        case {}, _, _:
            return (
                (b"{" if keywords != () else b"")
                + b" ".join(
                    dumps(
                        (k, v),  # ty: ignore[invalid-argument-type]
                        keywords=keywords,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                    if k is not None
                    else dumps(
                        v,  # ty: ignore[invalid-argument-type]
                        keywords=keywords,
                        format_=format_,
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                )
                + (b"}" if keywords != () else b"")
            )

        case float(), _common.FIELD_KEYWORDS, _:
            return b"uniform " + dumps(data, keywords=None, format_=format_)

        case np.ndarray(shape=(3,) | (6,) | (9,)), _common.FIELD_KEYWORDS, _:
            return b"uniform " + dumps(data.tolist(), keywords=None, format_=format_)  # ty: ignore[possibly-missing-attribute]

        case np.ndarray(shape=(_,)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<scalar> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 3)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<vector> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 6)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<symmTensor> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 9)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<tensor> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(), _, "binary":
            return (
                dumps(len(data), keywords=None, format_=None)  # ty: ignore[invalid-argument-type]
                + b"("
                + data.tobytes()  # ty: ignore[possibly-missing-attribute]
                + b")"
            )

        case np.ndarray(), (_, *_) | None, "ascii" | None:
            return dumps(len(data), keywords=None, format_=None) + dumps(  # ty: ignore[invalid-argument-type]
                data.tolist(),  # ty: ignore[possibly-missing-attribute]
                keywords=None,
                format_=format_,
            )

        case np.ndarray(), (), "ascii" | None:
            return dumps(data.tolist(), keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

        case DimensionSet(), _, _:
            return b"[" + dumps(tuple(data), keywords=None, format_=format_) + b"]"  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

        case Dimensioned(name=None), _, _:
            return (
                dumps(data.dimensions, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
                + b" "
                + dumps(data.value, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
            )  # ty: ignore[possibly-missing-attribute]

        case Dimensioned(name=str()), _, _:
            return (
                dumps(data.name, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
                + b" "
                + dumps(data.dimensions, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
                + b" "
                + dumps(data.value, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
            )

        case (
            tuple((_, _, *_)),
            _,
            _,
        ) if not isinstance(data, DimensionSet):
            if _tuple_is_keyword_entry:
                k, v = data  # ty: ignore[not-iterable]
                ret = b"\n" if isinstance(k, str) and k[0] == "#" else b""
                if k is not None:
                    ret += dumps(k, keywords=keywords)
                val = dumps(
                    v,  # ty: ignore[invalid-argument-type]
                    keywords=(*keywords, k)
                    if keywords is not None and k is not None
                    else ()
                    if k is None
                    else None,  # ty: ignore[invalid-argument-type]
                    format_=format_,
                )
                if k is not None and val:
                    ret += b" "
                ret += val
                if isinstance(k, str) and k[0] == "#":
                    ret += b"\n"
                elif k is not None and not isinstance(v, Mapping):
                    ret += b";"
                return ret

            return b" ".join(dumps(v, keywords=keywords, format_=format_) for v in data)  # ty: ignore[invalid-argument-type,not-iterable]

        case [*_], _, _:
            return (
                b"("
                + b" ".join(
                    dumps(
                        v,  # ty: ignore[invalid-argument-type]
                        keywords=None,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                    for v in data  #  ty: ignore[not-iterable]
                )
                + b")"
            )

        case None, _, _:
            return b""

        case True, _, _:
            return b"yes"

        case False, _, _:
            return b"no"

        case float() | int(), _, _:
            return dumps(str(data))

        case str(), _, _:
            return data.encode()  # ty: ignore[possibly-missing-attribute]

        case _:
            assert_never(data)  # ty: ignore[type-assertion-failure]
