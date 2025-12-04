from collections.abc import Mapping
from numbers import Integral, Real
from typing import overload
from warnings import warn

import numpy as np

from ._common import dict_from_items
from ._parsing import parse
from ._typing import (
    Data,
    DataEntry,
    DataEntryLike,
    DataLike,
    File,
    FileLike,
    KeywordEntry,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
)
from ._util import is_sequence
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
def normalize(
    data: FileLike,
    *,
    keywords: tuple[()],
    force_token: bool = False,
) -> File: ...


@overload
def normalize(
    data: DataLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    force_token: bool = False,
) -> Data: ...


@overload
def normalize(
    data: StandaloneDataLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    force_token: bool = False,
) -> StandaloneData: ...


@overload
def normalize(
    data: DataEntryLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    force_token: bool = False,
) -> DataEntry: ...


@overload
def normalize(
    data: SubDictLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    force_token: bool = False,
) -> SubDict: ...


def normalize(
    data: FileLike | DataLike | StandaloneDataLike | SubDictLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    force_token: bool = False,
) -> File | Data | StandaloneData | SubDict:
    match data, keywords, force_token:
        # File
        case Mapping(), (), False:
            items = (
                (
                    normalize(k, keywords=(), force_token=True)
                    if k is not None
                    else None,
                    normalize(
                        v,
                        keywords=(k,) if k is not None else (),  # ty: ignore[not-iterable]
                    ),  # ty: ignore[no-matching-overload]
                )
                for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
            )
            return dict_from_items(items, target=File)
        # Sub-dictionary
        case Mapping(), (_, *_), False:
            items = (
                (
                    normalize(k, keywords=keywords, force_token=True),  # ty: ignore[no-matching-overload]
                    normalize(v, keywords=(*keywords, k)),  # ty: ignore[no-matching-overload, not-iterable]
                )
                for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
            )
            return dict_from_items(items, target=SubDict)

        # Other dictionary
        case Mapping(), None, False:
            return dict_from_items(
                (
                    (
                        normalize(k, force_token=True),  # ty: ignore[no-matching-overload]
                        normalize(v),  # ty: ignore[no-matching-overload]
                    )
                    for k, v in data.items()  # ty: ignore[possibly-missing-attribute]
                )
            )

        # Numeric standalone data (n integers)
        case np.ndarray(shape=(_,), dtype=np.int64 | np.int32), (), False:
            return data  # ty: ignore[invalid-return-type]

        # Numeric standalone data (n x 3 floats)
        case np.ndarray(shape=(_,) | (_, 3), dtype=np.float64 | np.float32), (), False:
            return data  # ty: ignore[invalid-return-type]

        # Other possible numeric standalone data
        case [Real(), *_] | [
            [Real(), Real(), Real()],
            *_,
        ], (), False:
            try:
                return normalize(np.asarray(data), keywords=keywords)
            except ValueError:
                return normalize(data)  # ty: ignore[no-matching-overload]

        # Uniform field (scalar)
        case Real() | np.ndarray(
            shape=(), dtype=np.float64 | np.float32 | np.int64 | np.int32
        ), _, False if _expect_field(keywords):
            return float(data)  # ty: ignore[invalid-argument-type]

        # Uniform field (non-scalar)
        case np.ndarray(
            shape=(3,) | (6,) | (9,), dtype=np.float64 | np.float32
        ), _, False if _expect_field(keywords):
            return data  # ty: ignore[invalid-return-type]

        # Uniform field (non-scalar integer)
        case np.ndarray(
            shape=(3,) | (6,) | (9,), dtype=np.int64 | np.int32
        ), _, False if _expect_field(keywords):
            return normalize(data.astype(float), keywords=keywords)  # ty: ignore[possibly-missing-attribute]

        # Non-uniform field
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.float64 | np.float32
        ), _, False if _expect_field(keywords):
            return data  # ty: ignore[invalid-return-type]

        # Non-uniform field (integer)
        case np.ndarray(
            shape=(_,) | (_, 3) | (_, 6) | (_, 9), dtype=np.int64 | np.int32
        ), _, False if _expect_field(keywords):
            return normalize(data.astype(float), keywords=keywords)  # ty: ignore[possibly-missing-attribute]

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
        ], _, False if _expect_field(keywords) and not isinstance(data, tuple):
            try:
                return normalize(np.asarray(data), keywords=keywords)
            except ValueError:
                return [normalize(d) for d in data]  # ty: ignore[no-matching-overload,not-iterable]

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
        ], ("dimensions",), False:
            return DimensionSet(*data)  # ty: ignore[invalid-argument-type]

        # List
        case [*_], _, False if not isinstance(data, tuple):
            return [normalize(d) for d in data]  # ty: ignore[no-matching-overload,not-iterable]

        # Other Numpy array (treated as list)
        case np.ndarray(), _, False:
            return normalize(data.tolist())  # ty: ignore[possibly-missing-attribute]

        # Keyword entry
        case tuple((k, v)), None, False:
            return (normalize(k), normalize(v))  # ty: ignore[invalid-return-type]

        # Multiple data entries (tuple)
        case (
            tuple((_, _, *_)),
            _,
            False,
        ) if not isinstance(data, DimensionSet):
            ret = tuple(normalize(d, keywords=keywords) for d in data)  # ty: ignore[no-matching-overload,not-iterable]
            if any(isinstance(d, tuple) for d in ret):
                msg = f"Nested tuples not supported: {data!r}"
                raise ValueError(msg)
            return ret  # ty: ignore[invalid-return-type]

        # One-element tuple (unsupported)
        case tuple((_,)), _, False:
            msg = f"One-element tuple {data!r} not supported."
            raise ValueError(msg)

        # Empty tuple (unsupported)
        case tuple(()), _, False:
            msg = "Empty tuples are not supported"
            raise ValueError(msg)

        # Token
        case str(), _, True:
            if parse(data, target=str) != data:  # ty: ignore[invalid-argument-type]
                msg = f"{data!r} is not a valid keyword"
                raise ValueError(msg)
            return data  # ty: ignore[invalid-return-type]

        # Top-level string
        case str(), (), False:
            if not isinstance(parsed := parse(data, target=StandaloneData), str):  # ty: ignore[invalid-argument-type]
                msg = f"String {data!r} will be stored as {parsed!r}"
                warn(msg, stacklevel=2)
                return parsed
            return data  # ty: ignore[invalid-return-type]

        # String
        case str(), (_, *_) | None, False:
            if not isinstance(parsed := parse(data, target=Data), str):  # ty: ignore[invalid-argument-type]
                msg = f"String {data!r} will be stored as {parsed!r}"
                warn(msg, stacklevel=2)
                return parsed
            return data  # ty: ignore[invalid-return-type]

        # Boolean
        case True | False, _, False:
            return data  # ty: ignore[invalid-return-type]

        # Integer
        case Integral(), _, False:
            return int(data)  # ty: ignore[invalid-argument-type]

        # Float
        case Real(), _, False:
            return float(data)  # ty: ignore[invalid-argument-type]

        # Custom types
        case DimensionSet() | Dimensioned(), _, False:
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
    match data, keywords, header, tuple_is_keyword_entry:
        # Mapping
        case Mapping(), _, _, False:
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

        # Numpy array at top level (binary format)
        case np.ndarray(), (), _, False if (
            header.get("format", "") if header else ""  # ty: ignore[possibly-missing-attribute]
        ) == "binary":
            return dumps(len(data)) + b"(" + data.tobytes() + b")"  # ty: ignore[possibly-missing-attribute]

        # Numpy array at top level (non-binary)
        case np.ndarray(), (), _, False:
            return dumps(data.tolist())  # ty: ignore[possibly-missing-attribute]

        # Field data (scalar/vector/tensor)
        case int() | float() | np.ndarray(), _, _, False if _expect_field(keywords):
            data = np.asarray(data)
            class_ = header.get("class", "") if header else ""  # ty: ignore[possibly-missing-attribute]
            assert isinstance(class_, str)
            scalar = "Scalar" in class_

            shape = np.shape(data)
            if not shape or (not scalar and shape in ((3,), (6,), (9,))):
                return b"uniform " + dumps(data)

            # Non-uniform field
            assert isinstance(data, np.ndarray)
            ndim = np.ndim(data)
            if ndim == 1:
                tensor_kind = b"scalar"

            elif ndim == 2:
                assert len(shape) == 2
                if shape[1] == 3:
                    tensor_kind = b"vector"
                elif shape[1] == 6:
                    tensor_kind = b"symmTensor"
                elif shape[1] == 9:
                    tensor_kind = b"tensor"
                else:
                    return dumps(data)

            else:
                return dumps(data)

            binary = (header.get("format", "") if header else "") == "binary"  # ty: ignore[possibly-missing-attribute]

            contents = b"(" + data.tobytes() + b")" if binary else dumps(data)

            return b"nonuniform List<" + tensor_kind + b"> " + dumps(len(data)) + contents

        # DimensionSet
        case DimensionSet(), _, _, False:
            return b"[" + b" ".join(dumps(v) for v in data) + b"]"  # ty: ignore[not-iterable]

        # Dimensioned (with name)
        case Dimensioned(), _, _, False if data.name is not None:  # ty: ignore[possibly-missing-attribute]
            return (
                dumps(data.name)  # ty: ignore[possibly-missing-attribute]
                + b" "
                + dumps(data.dimensions)  # ty: ignore[possibly-missing-attribute]
                + b" "
                + dumps(data.value)  # ty: ignore[possibly-missing-attribute]
            )

        # Dimensioned (without name)
        case Dimensioned(), _, _, False:
            return dumps(data.dimensions) + b" " + dumps(data.value)  # ty: ignore[possibly-missing-attribute]

        # Tuple as keyword entry
        case tuple(), _, _, True:
            k, v = data  # ty: ignore[misc]
            ret = b"\n" if isinstance(k, str) and k[0] == "#" else b""
            if k is not None:
                ret += dumps(k, keywords=keywords)
            val = dumps(
                v,
                keywords=(*keywords, k)  # ty: ignore[not-iterable]
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

        # Tuple (not as keyword entry)
        case tuple(), _, _, False:
            return b" ".join(dumps(v, keywords=keywords, header=header) for v in data)  # ty: ignore[not-iterable]

        # Boolean True
        case True, _, _, False:
            return b"yes"

        # Boolean False
        case False, _, _, False:
            return b"no"

    # Sequence
    if is_sequence(data):
        return (
            b"(" + b" ".join(dumps(v, tuple_is_keyword_entry=True) for v in data) + b")"  # ty: ignore[not-iterable]
        )

    # Default case: convert to string
    return str(data).encode("latin-1")
