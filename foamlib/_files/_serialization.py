import array
import itertools
import sys
from enum import Enum, auto

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

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


def dumpb(
    data: FoamFileBase._SetData,
    *,
    kind: Kind = Kind.DEFAULT,
) -> bytes:
    if numpy and isinstance(data, np.ndarray):
        return dumpb(data.tolist(), kind=kind)

    if isinstance(data, Mapping):
        entries = []
        for k, v in data.items():
            b = dumpb(v, kind=kind)
            if isinstance(v, Mapping):
                entries.append(dumpb(k) + b" {" + b + b"}")
            elif not b:
                entries.append(dumpb(k) + b";")
            else:
                entries.append(dumpb(k) + b" " + b + b";")

        return b" ".join(entries)

    if isinstance(data, FoamFileBase.DimensionSet) or (
        kind == Kind.DIMENSIONS and is_sequence(data) and len(data) == 7
    ):
        return b"[" + b" ".join(dumpb(v) for v in data) + b"]"

    if (kind == Kind.FIELD or kind == Kind.BINARY_FIELD) and (
        isinstance(data, (int, float))
        or is_sequence(data)
        and data
        and isinstance(data[0], (int, float))
        and len(data) in (3, 6, 9)
    ):
        return b"uniform " + dumpb(data, kind=Kind.SINGLE_ENTRY)

    if (kind == Kind.FIELD or kind == Kind.BINARY_FIELD) and is_sequence(data):
        if isinstance(data[0], (int, float)):
            tensor_kind = b"scalar"
        elif len(data[0]) == 3:
            tensor_kind = b"vector"
        elif len(data[0]) == 6:
            tensor_kind = b"symmTensor"
        elif len(data[0]) == 9:
            tensor_kind = b"tensor"
        else:
            return dumpb(data)

        if kind == Kind.BINARY_FIELD:
            if tensor_kind == b"scalar":
                contents = b"(" + array.array("d", data).tobytes() + b")"
            else:
                contents = (
                    b"("
                    + array.array("d", itertools.chain.from_iterable(data)).tobytes()
                    + b")"
                )
        else:
            contents = dumpb(data, kind=Kind.SINGLE_ENTRY)

        return b"nonuniform List<" + tensor_kind + b"> " + dumpb(len(data)) + contents

    if kind != Kind.SINGLE_ENTRY and isinstance(data, tuple):
        return b" ".join(dumpb(v) for v in data)

    if isinstance(data, FoamFileBase.Dimensioned):
        if data.name is not None:
            return (
                dumpb(data.name)
                + b" "
                + dumpb(data.dimensions, kind=Kind.DIMENSIONS)
                + b" "
                + dumpb(data.value, kind=Kind.SINGLE_ENTRY)
            )
        return (
            dumpb(data.dimensions, kind=Kind.DIMENSIONS)
            + b" "
            + dumpb(data.value, kind=Kind.SINGLE_ENTRY)
        )

    if is_sequence(data):
        return b"(" + b" ".join(dumpb(v, kind=Kind.SINGLE_ENTRY) for v in data) + b")"

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
