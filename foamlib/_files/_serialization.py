from __future__ import annotations

import array
import itertools
import sys
from enum import Enum, auto
from typing import cast, overload

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

from ._parsing import DATA, TOKEN
from ._types import Data, Dimensioned, DimensionSet, Entry
from ._util import is_sequence

try:
    import numpy as np

    numpy = True
except ModuleNotFoundError:
    numpy = False


class Kind(Enum):
    DEFAULT = auto()
    KEYWORD = auto()
    SINGLE_ENTRY = auto()
    ASCII_FIELD = auto()
    DOUBLE_PRECISION_BINARY_FIELD = auto()
    SINGLE_PRECISION_BINARY_FIELD = auto()
    DIMENSIONS = auto()


@overload
def normalize(data: Data, *, kind: Kind = Kind.DEFAULT) -> Data: ...


@overload
def normalize(data: Entry, *, kind: Kind = Kind.DEFAULT) -> Entry: ...


def normalize(data: Entry, *, kind: Kind = Kind.DEFAULT) -> Entry:
    if numpy and isinstance(data, np.ndarray):
        ret = data.tolist()
        assert isinstance(ret, list)
        return ret

    if isinstance(data, Mapping):
        return {k: normalize(v, kind=kind) for k, v in data.items()}

    if (
        kind == Kind.DIMENSIONS
        and is_sequence(data)
        and len(data) <= 7
        and all(isinstance(d, (int, float)) for d in data)
    ):
        data = cast(Sequence[float], data)
        return DimensionSet(*data)

    if isinstance(data, tuple) and kind == Kind.SINGLE_ENTRY and len(data) == 2:
        k, v = data
        return (normalize(k, kind=Kind.KEYWORD), normalize(v))

    if is_sequence(data) and (kind == Kind.SINGLE_ENTRY or not isinstance(data, tuple)):
        return [normalize(d, kind=Kind.SINGLE_ENTRY) for d in data]

    if isinstance(data, Dimensioned):
        value = normalize(data.value, kind=Kind.SINGLE_ENTRY)
        assert isinstance(value, (int, float, list))
        return Dimensioned(value, data.dimensions, data.name)

    if isinstance(data, str):
        if kind == Kind.KEYWORD:
            data = TOKEN.parse_string(data, parse_all=True)[0]
            assert isinstance(data, str)
            return data

        return cast(Data, DATA.parse_string(data, parse_all=True)[0])

    if isinstance(
        data,
        (int, float, bool, tuple, DimensionSet),
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
        ret = dumps(k, kind=Kind.KEYWORD) + b" " + dumps(v)
        if not isinstance(v, Mapping):
            ret += b";"
        return ret

    if isinstance(data, DimensionSet):
        return b"[" + b" ".join(dumps(v) for v in data) + b"]"

    if kind in (
        Kind.ASCII_FIELD,
        Kind.DOUBLE_PRECISION_BINARY_FIELD,
        Kind.SINGLE_PRECISION_BINARY_FIELD,
    ) and (
        isinstance(data, (int, float))
        or (
            is_sequence(data)
            and data
            and isinstance(data[0], (int, float))
            and len(data) in (3, 6, 9)
        )
    ):
        return b"uniform " + dumps(data, kind=Kind.SINGLE_ENTRY)

    if kind in (
        Kind.ASCII_FIELD,
        Kind.DOUBLE_PRECISION_BINARY_FIELD,
        Kind.SINGLE_PRECISION_BINARY_FIELD,
    ) and is_sequence(data):
        if data and isinstance(data[0], (int, float)):
            tensor_kind = b"scalar"
        elif is_sequence(data[0]) and data[0] and isinstance(data[0][0], (int, float)):
            if len(data[0]) == 3:
                tensor_kind = b"vector"
            elif len(data[0]) == 6:
                tensor_kind = b"symmTensor"
            elif len(data[0]) == 9:
                tensor_kind = b"tensor"
            else:
                return dumps(data)
        else:
            return dumps(data)

        if kind in (
            Kind.DOUBLE_PRECISION_BINARY_FIELD,
            Kind.SINGLE_PRECISION_BINARY_FIELD,
        ):
            typecode = "f" if kind == Kind.SINGLE_PRECISION_BINARY_FIELD else "d"
            if tensor_kind == b"scalar":
                data = cast(Sequence[float], data)
                contents = b"(" + array.array(typecode, data).tobytes() + b")"
            else:
                data = cast(Sequence[Sequence[float]], data)
                contents = (
                    b"("
                    + array.array(
                        typecode, itertools.chain.from_iterable(data)
                    ).tobytes()
                    + b")"
                )
        else:
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
