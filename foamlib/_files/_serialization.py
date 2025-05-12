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
    Entry,
    EntryLike,
    is_sequence,
)


@overload
def normalize_data(
    data: DataLike, *, keywords: tuple[str, ...] | None = None
) -> Data: ...


@overload
def normalize_data(
    data: EntryLike, *, keywords: tuple[str, ...] | None = None
) -> Entry: ...


def normalize_data(
    data: EntryLike, *, keywords: tuple[str, ...] | None = None
) -> Entry:
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
        if is_sequence(data):
            try:
                arr = np.asarray(data)
            except ValueError:
                pass
            else:
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)

                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] in (3, 6, 9)):
                    return arr  # type: ignore [return-value]

            return [normalize_data(d) for d in data]

        if isinstance(data, int):
            return float(data)

        return normalize_data(data)

    if isinstance(data, Mapping):
        return {normalize_keyword(k): normalize_data(v) for k, v in data.items()}  # type: ignore [misc]

    if isinstance(data, np.ndarray):
        ret = data.tolist()
        assert isinstance(ret, (int, float, list))
        return ret

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
        return (
            normalize_keyword(k),
            normalize_data(v) if not isinstance(v, Mapping) else v,
        )  # type: ignore [return-value]

    if (
        is_sequence(data)
        and not isinstance(data, DimensionSet)
        and (keywords is None or not isinstance(data, tuple))
    ):
        return [normalize_data(d) for d in data]

    if isinstance(data, str):
        s = loads(data)
        if isinstance(s, (str, tuple, bool)):
            return s

    if isinstance(
        data,
        (int, float, bool, tuple, DimensionSet, Dimensioned),
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
    data: EntryLike,
    *,
    keywords: tuple[str, ...] | None = None,
    header: Mapping[str, Entry] | None = None,
    tuple_is_entry: bool = False,
) -> bytes:
    data = normalize_data(data, keywords=keywords)

    if isinstance(data, Mapping):
        return (
            b"{"
            + b" ".join(
                dumps(
                    (k, v),
                    keywords=keywords,
                    tuple_is_entry=True,
                )
                for k, v in data.items()
            )
            + b"}"
        )

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
        data = np.asarray(data)  # type: ignore [assignment]
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
        if tuple_is_entry:
            k, v = data
            ret = dumps(k)
            val = dumps(
                v,
                keywords=(*keywords, k)
                if keywords is not None and isinstance(k, str)
                else None,
            )
            if val:
                ret += b" " + val
            if not isinstance(v, Mapping):
                ret += b";"
            return ret

        return b" ".join(dumps(v) for v in data)

    if is_sequence(data):
        return b"(" + b" ".join(dumps(v, tuple_is_entry=True) for v in data) + b")"

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
