import contextlib
from collections.abc import Mapping
from typing import overload
from warnings import warn

import numpy as np

from ._parsing import Parsed
from ._typing import (
    Data,
    DataLike,
    File,
    FileLike,
    KeywordEntryLike,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
)
from ._util import add_to_mapping, as_dict_check_unique, is_sequence
from .types import Dimensioned, DimensionSet


def _expect_field(keywords: tuple[str, ...] | None) -> bool:
    return (
        keywords == ("internalField",)
        or keywords == ("internalField",)
        or (
            keywords is not None
            and len(keywords) == 3
            and keywords[0] == "boundaryField"
            and (
                keywords[2] == "value"
                or keywords[2] == "gradient"
                or keywords[2].endswith(("Value", "Gradient"))
            )
        )
    )


@overload
def normalize(
    data: FileLike,
    *,
    keywords: tuple[()],
    bool_ok: bool = True,
) -> File: ...


@overload
def normalize(
    data: DataLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> Data: ...


@overload
def normalize(
    data: StandaloneDataLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> StandaloneData: ...


@overload
def normalize(
    data: SubDictLike,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> SubDict: ...


@overload
def normalize(
    data: None,
    /,
    *,
    keywords: tuple[()],
    bool_ok: bool = True,
) -> None: ...


def normalize(
    data: FileLike | DataLike | StandaloneDataLike | SubDictLike | None,
    /,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> File | Data | StandaloneData | SubDict | None:
    match data:
        # Dictionary
        case Mapping():
            items = (
                (
                    normalize(k, keywords=keywords, bool_ok=False),
                    normalize(
                        v, keywords=(*keywords, k) if keywords is not None else None
                    ),
                )
                for k, v in data.items()
            )
            if keywords is None:
                return as_dict_check_unique(items)
            ret1: SubDict = {}
            for k, v in items:
                if k is not None and k.startswith("#"):
                    if isinstance(v, Mapping):
                        msg = f"Directive {k} cannot have a dictionary as value"
                        raise ValueError(msg)
                elif k in ret1:
                    msg = (
                        f"Duplicate keyword {k} in dictionary with keywords {keywords}"
                    )
                    raise ValueError(msg)
                ret1 = add_to_mapping(ret1, k, v)
            return ret1

        # Numeric standalone data (n integers)
        case np.ndarray(shape=(_,), dtype=dtype) if keywords == () and np.issubdtype(
            dtype, np.integer):
            return data

        # Numeric standalone data (n x 3 floats)
        case np.ndarray(shape=(_,) | (_, 3)) if keywords == () and np.issubdtype(
            data.dtype, np.floating):
            return data

        # Other possible numeric standalone data
        case [*_] if keywords == () and not isinstance(data, tuple):
            try:
                return normalize(np.asarray(data), keywords=keywords, bool_ok=bool_ok)
            except ValueError:
                return normalize(data, bool_ok=bool_ok)

        # Uniform field (scalar)
        case float() | int() | np.ndarray(shape=()) if _expect_field(keywords):
            return float(data)
        
        # Uniform field (non-scalar)
        case np.ndarray(shape=(3,) | (6,) | (9,)) if _expect_field(keywords):
            if not np.issubdtype(data.dtype, np.floating):
                return data.astype(float)
            return data
        
        # Non-uniform field
        case np.ndarray(shape=(_,) | ( _, 3) | (_, 6) | (_, 9)) if _expect_field(keywords):
            if not np.issubdtype(data.dtype, np.floating):
                return data.astype(float)
            return data
        
        # Other possible field
        case [*_] if _expect_field(keywords) and not isinstance(data, tuple):
            try:
                return normalize(np.asarray(data, dtype=float), keywords=keywords, bool_ok=bool_ok)
            except ValueError:
                return [normalize(d, bool_ok=bool_ok) for d in data]
            
        # Dimension set from list of numbers
        case [*_] if keywords == ("dimensions",) and all(
            isinstance(d, (int, float)) for d in data
        ) and len(data) <= 7:
            return DimensionSet(*data)
        
        # List
        case [*_] if not isinstance(data, tuple):
            return [normalize(d, bool_ok=bool_ok) for d in data]
        
        # Other Numpy array (treated as list)
        case np.ndarray():
            return normalize(data.tolist(), bool_ok=bool_ok)
        
        # Keyword entry
        case tuple((k, v)) if keywords is None:
            return (normalize(k, bool_ok=False), normalize(v))

        # Multiple data entries (tuple)
        case tuple((_, _, *_)) if not isinstance(data, DimensionSet):
            return tuple(normalize(d, keywords=keywords, bool_ok=bool_ok) for d in data)
        
        # Single data entry in a tuple
        case tuple((d,)) if not isinstance(data, DimensionSet):
            msg = f"One-element tuple {data!r} will be retrieved as {d!r} {type(d)}"
            warn(msg)
            return normalize(d, keywords=keywords, bool_ok=bool_ok)

        # Empty tuple (unsupported)
        case tuple(()) if not isinstance(data, DimensionSet):
            msg = "Empty tuples are not supported"
            raise ValueError(msg)

        # String
        case str():
            parsed = Parsed(data)[()]
            if not bool_ok and isinstance(data, bool):
                return data

            if not isinstance(parsed, str):
                msg = f"String {data!r} will be retrieved as {parsed!r} {type(parsed)}"
                warn(msg)

            return parsed
        
        # None (only valid as top-level dictionary key)
        case None if keywords == ():
            return data
        
        # Boolean
        case True | False if bool_ok:
            return data
        
        # Other basic types
        case int() | float() | DimensionSet() | Dimensioned():
            return data

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


"""
    if isinstance(data, Mapping):
        items = (
            (
                normalize(k, keywords=keywords, bool_ok=False),  # ty: ignore[no-matching-overload]
                normalize(v, keywords=(*keywords, k) if keywords is not None else None),  # ty: ignore[no-matching-overload]
            )
            for k, v in data.items()
        )
        if keywords is None:
            return as_dict_check_unique(items)
        ret1: SubDict = {}
        for k, v in items:
            if k is not None and k.startswith("#"):
                if isinstance(v, Mapping):
                    msg = f"Directive {k} cannot have a dictionary as value"
                    raise ValueError(msg)
            elif k in ret1:
                msg = f"Duplicate keyword {k} in dictionary with keywords {keywords}"
                raise ValueError(msg)
            ret1 = add_to_mapping(ret1, k, v)  # ty: ignore[invalid-assignment]
        return ret1

    if keywords == () and is_sequence(data) and not isinstance(data, tuple):
        try:
            arr = np.asarray(data)
        except ValueError:
            pass
        else:
            if np.issubdtype(arr.dtype, np.integer) and arr.ndim == 1:
                return arr
            if arr.ndim == 2 and arr.shape[1] == 3:
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                return arr

    if keywords is not None and (
        keywords == ("internalField",)
        or (
            len(keywords) == 3
            and keywords[0] == "boundaryField"
            and (
                keywords[2] == "value"
                or keywords[2] == "gradient"
                or keywords[2].endswith(("Value", "Gradient"))
            )
        )
    ):
        if is_sequence(data) and not isinstance(data, tuple):
            try:
                arr = np.asarray(data)
            except ValueError:
                pass
            else:
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)

                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] in (3, 6, 9)):
                    return arr

            return [normalize(d) for d in data]  # ty: ignore[not-iterable]

        if isinstance(data, int):
            return float(data)

        return normalize(data)  # ty: ignore[no-matching-overload]

    if isinstance(data, np.ndarray):
        if keywords == (None,):
            return data  # ty: ignore[invalid-return-type]
        ret2 = data.tolist()
        assert isinstance(ret2, (int, float, list))
        return ret2

    if (
        not isinstance(data, DimensionSet)
        and keywords is not None
        and keywords == ("dimensions",)
        and is_sequence(data)
        and len(data) <= 7  # ty: ignore[invalid-argument-type]
        and all(isinstance(d, (int, float)) for d in data)  # ty: ignore[not-iterable]
    ):
        return DimensionSet(*data)  # ty: ignore[invalid-argument-type]

    if isinstance(data, tuple) and not isinstance(data, DimensionSet):
        if tuple_is_keyword_entry:
            k2, v2 = data
            if isinstance(k2, Mapping):
                msg = "Keyword in keyword entry cannot be a dictionary"
                raise ValueError(msg)
            k2 = normalize(k2, keywords=keywords, bool_ok=False)  # ty: ignore[no-matching-overload]
            v2 = normalize(
                v2, keywords=(*keywords, k2) if keywords is not None else keywords
            )  # ty: ignore[no-matching-overload]
            return (k2, v2)
        return tuple(normalize(d, keywords=keywords) for d in data)  # ty: ignore[no-matching-overload]

    if (
        is_sequence(data)
        and not isinstance(data, DimensionSet)
        and not isinstance(data, tuple)
    ):
        return [normalize(d) for d in data]  # ty: ignore[not-iterable]

    if isinstance(data, str):
        with contextlib.suppress(ValueError, KeyError):
            s = Parsed(data)[()]
            if not bool_ok and isinstance(s, bool):
                return data
            if isinstance(s, (str, tuple, bool)):
                return s

    if isinstance(
        data,
        (int, float, bool, DimensionSet, Dimensioned),
    ):
        return data

    if data is None:
        if keywords != ():
            msg = "Unexpected None key in subdictionary"
            raise TypeError(msg)
        return data

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)
"""


def dumps(
    data: FileLike | DataLike | StandaloneDataLike | KeywordEntryLike | SubDictLike,
    *,
    keywords: tuple[str, ...] | None = None,
    header: SubDictLike | None = None,
    tuple_is_keyword_entry: bool = False,
) -> bytes:
    data = normalize(  # type: ignore[no-matching-overload]
        data, keywords=keywords
    )

    if isinstance(data, Mapping):
        return (
            (b"{" if keywords != () else b"")
            + b" ".join(
                dumps(
                    (k, v),
                    keywords=keywords,
                    header=header,
                    tuple_is_keyword_entry=True,
                )
                for k, v in data.items()
            )
            + (b"}" if keywords != () else b"")
        )

    if keywords == (None,) and isinstance(data, np.ndarray):
        if (header.get("format", "") if header else "") == "binary":
            return dumps(len(data)) + b"(" + data.tobytes() + b")"

        return dumps(data.tolist())

    if (
        keywords is not None
        and (
            keywords == ("internalField",)
            or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and (
                    keywords[2] == "value"
                    or keywords[2] == "gradient"
                    or keywords[2].endswith(("Value", "Gradient"))
                )
            )
        )
        and isinstance(data, (int, float, np.ndarray))
    ):
        data = np.asarray(data)
        class_ = header.get("class", "") if header else ""
        assert isinstance(class_, str)
        scalar = "Scalar" in class_

        shape = np.shape(data)
        if not shape or (not scalar and shape in ((3,), (6,), (9,))):
            return b"uniform " + dumps(data)

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

        binary = (header.get("format", "") if header else "") == "binary"

        contents = b"(" + data.tobytes() + b")" if binary else dumps(data)

        return b"nonuniform List<" + tensor_kind + b"> " + dumps(len(data)) + contents

    if isinstance(data, DimensionSet):
        return b"[" + b" ".join(dumps(v) for v in data) + b"]"

    if isinstance(data, Dimensioned):
        if data.name is not None:
            return (
                dumps(data.name)
                + b" "
                + dumps(data.dimensions)
                + b" "
                + dumps(data.value)
            )
        return dumps(data.dimensions) + b" " + dumps(data.value)

    if isinstance(data, tuple):
        if tuple_is_keyword_entry:
            k, v = data
            ret = b"\n" if isinstance(k, str) and k[0] == "#" else b""
            if k is not None:
                ret += dumps(k, keywords=keywords)
            val = dumps(
                v,
                keywords=(*keywords, k) if keywords is not None else None,
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

        return b" ".join(dumps(v, keywords=keywords, header=header) for v in data)

    if is_sequence(data):
        return (
            b"(" + b" ".join(dumps(v, tuple_is_keyword_entry=True) for v in data) + b")"  # ty: ignore[not-iterable]
        )

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
