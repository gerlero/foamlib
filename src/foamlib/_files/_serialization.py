from collections.abc import Mapping
from numbers import Integral, Real
from typing import overload
from warnings import warn

import numpy as np

from ._common import dict_from_items
from ._parsing import parse
from ._typing import (
    Data,
    DataLike,
    Dict,
    DictLike,
    File,
    FileLike,
    KeywordEntry,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
)
from .types import Dimensioned, DimensionSet


def _expect_field(keywords: tuple[str, ...] | None) -> bool:
    match keywords:
        case ("internalField",):
            return True
        case ("boundaryField", _, kw) if kw in ("value", "gradient") or kw.endswith(
            ("Value", "Gradient")
        ):
            return True

    return False


@overload
def normalized(
    data: FileLike,
    *,
    keywords: tuple[()] = ...,
) -> File: ...


@overload
def normalized(
    data: DataLike,
    /,
    *,
    keywords: tuple[str, ...] | None = ...,
) -> Data: ...


@overload
def normalized(
    data: StandaloneDataLike,
    /,
    *,
    keywords: tuple[()] = ...,
) -> StandaloneData: ...


@overload
def normalized(
    data: SubDictLike,
    /,
    *,
    keywords: tuple[str, ...] = ...,
) -> SubDict: ...


@overload
def normalized(
    data: DataLike,
    /,
    *,
    keywords: tuple[str, ...] = ...,
) -> Data: ...


@overload
def normalized(
    data: DictLike,
    /,
    *,
    keywords: None = ...,
) -> Data: ...


@overload
def normalized(
    data: None,
    /,
    *,
    keywords: tuple[str, ...] = ...,
) -> None: ...


def normalized(
    data: FileLike | DataLike | StandaloneDataLike | SubDictLike | DictLike | None,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
) -> File | Data | StandaloneData | SubDict | Dict | None:
    match data, keywords:
        # File
        case Mapping(), ():
            return dict_from_items(
                (
                    (
                        k,
                        normalized(
                            v,
                            keywords=(k,) if k is not None else (),  # ty: ignore[not-iterable]
                        ),  # ty: ignore[no-matching-overload]
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                ),
                target=File,
                check_keys=True,
            )  # ty: ignore[invalid-return-type]

        # Sub-dictionary
        case Mapping(), (_, *_):
            return dict_from_items(
                (
                    (
                        k,
                        normalized(v, keywords=(*keywords, k)),  # ty: ignore[no-matching-overload, not-iterable]
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                ),
                target=SubDict,
                check_keys=True,
            )  # ty: ignore[invalid-return-type]

        # Other dictionary
        case Mapping(), None:
            return dict_from_items(
                (
                    (
                        k,
                        normalized(v),  # ty: ignore[no-matching-overload]
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                ),
                target=Dict,
                check_keys=True,
            )  # ty: ignore[invalid-return-type]

        # Numeric standalone data (n integers)
        case np.ndarray(shape=(_,), dtype=np.int64 | np.int32), ():
            return data  # ty: ignore[invalid-return-type]

        # Numeric standalone data (n x 3 floats)
        case np.ndarray(shape=(_,) | (_, 3), dtype=np.float64 | np.float32), ():
            return data  # ty: ignore[invalid-return-type]

        # Other possible numeric standalone data
        case [Real(), *_] | [
            [Real(), Real(), Real()],
            *_,
        ], ():
            try:
                return normalized(np.asarray(data), keywords=keywords)
            except ValueError:
                return normalized(data)  # ty: ignore[no-matching-overload]

        # Uniform field (scalar)
        case Real() | np.ndarray(
            shape=(), dtype=np.float64 | np.float32 | np.int64 | np.int32
        ), _ if _expect_field(keywords):
            return float(data)  # ty: ignore[invalid-argument-type]

        # Uniform field (non-scalar)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.float64 | np.float32), _ if (
            _expect_field(keywords)
        ):
            return data  # ty: ignore[invalid-return-type]

        # Uniform field (non-scalar integer)
        case np.ndarray(shape=(3,) | (6,) | (9,), dtype=np.int64 | np.int32), _ if (
            _expect_field(keywords)
        ):
            return normalized(data.astype(float), keywords=keywords)  # ty: ignore[possibly-missing-attribute]

        # Non-uniform field
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.float64 | np.float32
        ), _ if _expect_field(keywords):
            return data  # ty: ignore[invalid-return-type]

        # Non-uniform field (integer)
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.int64 | np.int32
        ), _ if _expect_field(keywords):
            return normalized(data.astype(float), keywords=keywords)  # ty: ignore[possibly-missing-attribute]

        # Other possible field
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
        ] | [Real(), *_] | [
            [Real(), Real(), Real()],
            *_,
        ] | [
            [
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
            ],
            *_,
        ] | [
            [
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
                Real(),
            ],
            *_,
        ], _ if _expect_field(keywords) and not isinstance(data, tuple):
            try:
                return normalized(np.asarray(data), keywords=keywords)
            except ValueError:
                return [normalized(d) for d in data]  # ty: ignore[no-matching-overload,not-iterable]

        # Dimension set from list of numbers
        case [] | [Real()] | [Real(), Real()] | [
            Real(),
            Real(),
            Real(),
        ] | [Real(), Real(), Real(), Real()] | [
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
        ] | [
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
            Real(),
        ], ("dimensions",):
            return DimensionSet(*data)  # ty: ignore[invalid-argument-type]

        # List
        case [*_], _ if not isinstance(data, tuple):
            return [normalized(d) for d in data]  # ty: ignore[no-matching-overload,not-iterable]

        # Other Numpy array (treated as list)
        case np.ndarray(), _:
            return normalized(data.tolist())  # ty: ignore[possibly-missing-attribute]

        # Multiple data entries or keyword entry (tuple)
        case (
            tuple((_, _, *_)),
            _,
        ) if not isinstance(data, DimensionSet):
            ret = tuple(normalized(d, keywords=keywords) for d in data)  # ty: ignore[no-matching-overload,not-iterable]
            if any(isinstance(d, tuple) for d in ret):
                msg = f"Nested tuples not supported: {data!r}"
                raise ValueError(msg)
            return ret  # ty: ignore[invalid-return-type]

        # One-element tuple (unsupported)
        case tuple((_,)), _:
            msg = f"One-element tuple {data!r} not supported"
            raise ValueError(msg)

        # Empty tuple (unsupported)
        case tuple(()), _:
            msg = "Empty tuples are not supported"
            raise ValueError(msg)

        # Top-level string
        case str(), ():
            match parsed := parse(data, target=StandaloneData):  # ty: ignore[invalid-argument-type]
                case str():
                    if not parsed:
                        msg = "Found unsupported empty string"
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

        # String
        case str(), (_, *_) | None:
            match parsed := parse(data, target=Data):  # ty: ignore[invalid-argument-type]
                case str():
                    if not parsed:
                        msg = "Found unsupported empty string"
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

        # None
        case None, (*_,):
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

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def dumps(
    data: File | Data | StandaloneData | KeywordEntry | SubDict,
    *,
    keywords: tuple[str, ...] | None = None,
    header: SubDictLike | None = None,
    tuple_is_keyword_entry: bool = False,
) -> bytes:
    if keywords == () and header is None and isinstance(data, Mapping):
        header = data.get("FoamFile")  # ty: ignore[invalid-argument-type,invalid-assignment]

    match data, header:
        case Mapping(), _:
            return (
                (b"{" if keywords != () else b"")
                + b" ".join(
                    dumps(
                        (k, v),  # ty: ignore[invalid-argument-type]
                        keywords=keywords,
                        header=header,
                        tuple_is_keyword_entry=True,
                    )
                    if k is not None
                    else dumps(
                        v,  # ty: ignore[invalid-argument-type]
                        keywords=keywords,
                        header=header,
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                )
                + (b"}" if keywords != () else b"")
            )

        case np.ndarray(), {"format": "binary"} if not _expect_field(keywords):
            return dumps(len(data)) + b"(" + data.tobytes() + b")"  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

        case np.ndarray(), _ if not _expect_field(keywords):
            return dumps(len(data)) + dumps(data.tolist())  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

        case float(), _ if _expect_field(keywords):
            return b"uniform " + dumps(data)

        case np.ndarray(shape=(3,) | (6,) | (9,)), _ if _expect_field(keywords):
            return b"uniform " + dumps(data.tolist())  # ty: ignore[possibly-missing-attribute]

        case np.ndarray(shape=(_,)), _ if _expect_field(keywords):
            return b"nonuniform List<scalar> " + dumps(data, header=header)

        case np.ndarray(shape=(_, 3)), _ if _expect_field(keywords):
            return b"nonuniform List<vector> " + dumps(data, header=header)

        case np.ndarray(shape=(_, 6)), _ if _expect_field(keywords):
            return b"nonuniform List<symmTensor> " + dumps(data, header=header)

        case np.ndarray(shape=(_, 9)), _ if _expect_field(keywords):
            return b"nonuniform List<tensor> " + dumps(data, header=header)

        case DimensionSet(), _:
            return b"[" + dumps(tuple(data)) + b"]"  # ty: ignore[invalid-argument-type,possibly-missing-attribute]

        case Dimensioned(name=None), _:
            return dumps(data.dimensions) + b" " + dumps(data.value)  # ty: ignore[possibly-missing-attribute]

        case Dimensioned(), _:
            return (
                dumps(data.name)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
                + b" "
                + dumps(data.dimensions)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
                + b" "
                + dumps(data.value)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
            )

        case tuple((_, _, *_)), _ if not isinstance(data, DimensionSet):
            if tuple_is_keyword_entry:
                k, v = data  # type: ignore[not-iterable]
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
                    header=header,
                )
                if k is not None and val:
                    ret += b" "
                ret += val
                if isinstance(k, str) and k[0] == "#":
                    ret += b"\n"
                elif k is not None and not isinstance(v, Mapping):
                    ret += b";"
                return ret

            return b" ".join(dumps(v, keywords=keywords, header=header) for v in data)  # ty: ignore[invalid-argument-type,not-iterable]

        case [*_], _:
            return (
                b"("
                + b" ".join(dumps(v, tuple_is_keyword_entry=True) for v in data)  # ty: ignore[invalid-argument-type,not-iterable]
                + b")"
            )

        case None, _:
            return b""

        case True, _:
            return b"yes"

        case False, _:
            return b"no"

        case _:
            return str(data).encode("latin-1")
