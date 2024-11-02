from __future__ import annotations

import array
import itertools
import sys
from enum import Enum, auto
from typing import cast

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

from ._base import FoamFileBase
from ._util import is_sequence

try:
    import numpy as np

    numpy = True
except ModuleNotFoundError:
    numpy = False


class Kind(Enum):
    DEFAULT = auto()
    SINGLE_ENTRY = auto()
    FIELD = auto()
    BINARY_FIELD = auto()
    DIMENSIONS = auto()


def dumps(
    data: FoamFileBase.Data,
    *,
    kind: Kind = Kind.DEFAULT,
) -> bytes:
    if numpy and isinstance(data, np.ndarray):
        return dumps(data.tolist(), kind=kind)

    if isinstance(data, Mapping):
        entries = []
        for k, v in data.items():
            b = dumps(v, kind=kind)
            if isinstance(v, Mapping):
                entries.append(dumps(k) + b" {" + b + b"}")
            elif not b:
                entries.append(dumps(k) + b";")
            else:
                entries.append(dumps(k) + b" " + b + b";")

        return b" ".join(entries)

    if isinstance(data, FoamFileBase.DimensionSet) or (
        kind == Kind.DIMENSIONS and is_sequence(data) and len(data) == 7
    ):
        return b"[" + b" ".join(dumps(v) for v in data) + b"]"

    if kind in (Kind.FIELD, Kind.BINARY_FIELD) and (
        isinstance(data, (int, float))
        or is_sequence(data)
        and data
        and isinstance(data[0], (int, float))
        and len(data) in (3, 6, 9)
    ):
        return b"uniform " + dumps(data, kind=Kind.SINGLE_ENTRY)

    if kind in (Kind.FIELD, Kind.BINARY_FIELD) and is_sequence(data):
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

        if kind == Kind.BINARY_FIELD:
            if tensor_kind == b"scalar":
                data = cast(Sequence[float], data)
                contents = b"(" + array.array("d", data).tobytes() + b")"
            else:
                data = cast(Sequence[Sequence[float]], data)
                contents = (
                    b"("
                    + array.array("d", itertools.chain.from_iterable(data)).tobytes()
                    + b")"
                )
        else:
            contents = dumps(data, kind=Kind.SINGLE_ENTRY)

        return b"nonuniform List<" + tensor_kind + b"> " + dumps(len(data)) + contents

    if kind != Kind.SINGLE_ENTRY and isinstance(data, tuple):
        return b" ".join(dumps(v) for v in data)

    if isinstance(data, FoamFileBase.Dimensioned):
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

    if is_sequence(data):
        return b"(" + b" ".join(dumps(v, kind=Kind.SINGLE_ENTRY) for v in data) + b")"

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
