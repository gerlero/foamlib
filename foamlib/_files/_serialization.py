from __future__ import annotations

import sys
from typing import overload

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

import numpy as np

from ._parsing import loads
from ._types import (
    Data,
    DataLike,
    Dimensioned,
    DimensionSet,
    KeywordEntryLike,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
    SubDictLike,
    is_sequence,
)


@overload
def normalize_data(
    data: DataLike, *, keywords: tuple[str, ...] | None = None
) -> Data: ...


@overload
def normalize_data(
    data: StandaloneDataLike, *, keywords: tuple[str, ...] | None = None
) -> StandaloneData: ...


@overload
def normalize_data(
    data: SubDictLike, *, keywords: tuple[str, ...] | None = None
) -> SubDict: ...


def normalize_data(
    data: DataLike | StandaloneDataLike | SubDictLike,
    *,
    keywords: tuple[str, ...] | None = None,
) -> Data | StandaloneData | SubDict:
    if isinstance(data, Mapping):
        return {normalize_keyword(k): normalize_data(v) for k, v in data.items()}  # type: ignore [arg-type, misc]

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

            return [normalize_data(d) for d in data]  # type: ignore [arg-type, return-value]

        if isinstance(data, int):
            return float(data)

        return normalize_data(data)

    if isinstance(data, np.ndarray):
        ret = data.tolist()
        assert isinstance(ret, (int, float, list))
        return ret  # type: ignore [return-value]

    if (
        not isinstance(data, DimensionSet)
        and keywords is not None
        and keywords == ("dimensions",)
        and is_sequence(data)
        and len(data) <= 7
        and all(isinstance(d, (int, float)) for d in data)
    ):
        return DimensionSet(*data)

    if keywords is None and isinstance(data, tuple) and len(data) == 2:
        k, v = data
        assert not isinstance(k, Mapping)
        return (  # type: ignore [return-value]
            normalize_keyword(k),  # type: ignore [arg-type]
            normalize_data(v) if not isinstance(v, Mapping) else v,  # type: ignore [arg-type]
        )

    if (
        is_sequence(data)
        and not isinstance(data, DimensionSet)
        and not isinstance(data, tuple)
    ):
        return [normalize_data(d) for d in data]  # type: ignore [arg-type, return-value]

    if isinstance(data, tuple) and not isinstance(data, DimensionSet):
        return tuple(normalize_data(d, keywords=keywords) for d in data)  # type: ignore [misc]

    if isinstance(data, str):
        s = loads(data, keywords=keywords)
        if isinstance(s, (str, tuple, bool)):
            return s

    if isinstance(
        data,
        (int, float, bool, DimensionSet, Dimensioned),
    ):
        return data

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def normalize_keyword(data: DataLike) -> Data:
    ret = normalize_data(data)

    if isinstance(data, str) and isinstance(ret, bool):
        return data

    return ret


def dumps(
    data: DataLike | StandaloneDataLike | KeywordEntryLike | SubDictLike,
    *,
    keywords: tuple[str, ...] | None = None,
    header: SubDictLike | None = None,
    tuple_is_keyword_entry: bool = False,
) -> bytes:
    data = normalize_data(data, keywords=keywords)  # type: ignore [arg-type, misc]

    if isinstance(data, Mapping):
        return (
            b"{"
            + b" ".join(
                dumps(
                    (k, v),
                    keywords=keywords,
                    tuple_is_keyword_entry=True,
                )
                for k, v in data.items()
            )
            + b"}"
        )

    if keywords == () and isinstance(data, np.ndarray):
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
            ret += dumps(k)
            val = dumps(
                v,
                keywords=(*keywords, k)
                if keywords is not None and isinstance(k, str)
                else None,
            )
            if val:
                ret += b" " + val
            if isinstance(k, str) and k[0] == "#":
                ret += b"\n"
            elif not isinstance(v, Mapping):
                ret += b";"
            return ret

        return b" ".join(dumps(v, keywords=keywords, header=header) for v in data)

    if is_sequence(data):
        return (
            b"(" + b" ".join(dumps(v, tuple_is_keyword_entry=True) for v in data) + b")"
        )

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
