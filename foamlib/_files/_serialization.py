from __future__ import annotations

import sys
from enum import Enum, auto
from typing import overload

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

import numpy as np

from ._parsing import parse_data
from ._types import Data, Dimensioned, DimensionSet, Entry, is_sequence


class Kind(Enum):
    DEFAULT = auto()
    SINGLE_ENTRY = auto()
    ASCII_FIELD = auto()
    SCALAR_ASCII_FIELD = auto()
    BINARY_FIELD = auto()
    SCALAR_BINARY_FIELD = auto()
    DIMENSIONS = auto()


@overload
def normalize(data: Data, *, kind: Kind = Kind.DEFAULT) -> Data: ...


@overload
def normalize(data: Entry, *, kind: Kind = Kind.DEFAULT) -> Entry: ...


def normalize(data: Entry, *, kind: Kind = Kind.DEFAULT) -> Entry:
    if kind in (
        Kind.ASCII_FIELD,
        Kind.SCALAR_ASCII_FIELD,
        Kind.BINARY_FIELD,
        Kind.SCALAR_BINARY_FIELD,
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

            return data

        if isinstance(data, int):
            return float(data)

        return data

    if isinstance(data, np.ndarray):
        ret = data.tolist()
        assert isinstance(ret, (int, float, list))
        return ret

    if isinstance(data, Mapping):
        return {k: normalize(v, kind=kind) for k, v in data.items()}

    if (
        kind == Kind.DIMENSIONS
        and is_sequence(data)
        and len(data) <= 7
        and all(isinstance(d, (int, float)) for d in data)
    ):
        return DimensionSet(*data)

    if isinstance(data, tuple) and kind == Kind.SINGLE_ENTRY and len(data) == 2:
        k, v = data
        return (normalize(k), normalize(v))

    if is_sequence(data) and (kind == Kind.SINGLE_ENTRY or not isinstance(data, tuple)):
        return [normalize(d, kind=Kind.SINGLE_ENTRY) for d in data]

    if isinstance(data, str):
        return parse_data(data)

    if isinstance(
        data,
        (int, float, bool, tuple, DimensionSet, Dimensioned),
    ):
        return data

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def dumps(
    data: Entry,
    *,
    kind: Kind = Kind.DEFAULT,
) -> bytes:
    data = normalize(data, kind=kind)

    if isinstance(data, Mapping):
        return (
            b"{"
            + b" ".join(dumps((k, v), kind=Kind.SINGLE_ENTRY) for k, v in data.items())
            + b"}"
        )

    if isinstance(data, tuple) and kind == Kind.SINGLE_ENTRY and len(data) == 2:
        k, v = data
        ret = dumps(k) + b" " + dumps(v)
        if not isinstance(v, Mapping):
            ret += b";"
        return ret

    if isinstance(data, DimensionSet):
        return b"[" + b" ".join(dumps(v) for v in data) + b"]"

    if kind in (
        Kind.ASCII_FIELD,
        Kind.SCALAR_ASCII_FIELD,
        Kind.BINARY_FIELD,
        Kind.SCALAR_BINARY_FIELD,
    ) and (isinstance(data, (int, float, np.ndarray))):
        shape = np.shape(data)
        if not shape or (
            kind not in (Kind.SCALAR_ASCII_FIELD, Kind.SCALAR_BINARY_FIELD)
            and shape in ((3,), (6,), (9,))
        ):
            return b"uniform " + dumps(data, kind=Kind.SINGLE_ENTRY)

        assert isinstance(data, np.ndarray)
        ndim = len(shape)
        if ndim == 1:
            tensor_kind = b"scalar"

        elif ndim == 2:
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

        if kind in (Kind.BINARY_FIELD, Kind.SCALAR_BINARY_FIELD):
            contents = b"(" + data.tobytes() + b")"
        else:
            assert kind in (Kind.ASCII_FIELD, Kind.SCALAR_ASCII_FIELD)
            contents = dumps(data, kind=Kind.SINGLE_ENTRY)

        return b"nonuniform List<" + tensor_kind + b"> " + dumps(len(data)) + contents

    if isinstance(data, Dimensioned):
        if data.name is not None:
            return (
                dumps(data.name)
                + b" "
                + dumps(data.dimensions, kind=Kind.DIMENSIONS)
                + b" "
                + dumps(data.value, kind=Kind.SINGLE_ENTRY)
            )
        return (
            dumps(data.dimensions, kind=Kind.DIMENSIONS)
            + b" "
            + dumps(data.value, kind=Kind.SINGLE_ENTRY)
        )

    if isinstance(data, tuple):
        return b" ".join(dumps(v) for v in data)

    if is_sequence(data) and not isinstance(data, tuple):
        return b"(" + b" ".join(dumps(v, kind=Kind.SINGLE_ENTRY) for v in data) + b")"

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
