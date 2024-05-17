import array
import itertools
import sys
from enum import Enum, auto

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

from .._util import is_sequence
from ._base import FoamDict

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
    data: FoamDict._SetData,
    *,
    kind: Kind = Kind.DEFAULT,
) -> bytes:
    if numpy and isinstance(data, np.ndarray):
        return dumpb(data.tolist(), kind=kind)

    elif isinstance(data, Mapping):
        entries = []
        for k, v in data.items():
            b = dumpb(v, kind=kind)
            if not k:
                entries.append(b)
            elif isinstance(v, Mapping):
                entries.append(dumpb(k) + b"\n" + b"{\n" + b + b"\n}")
            elif not b:
                entries.append(dumpb(k) + b";")
            else:
                entries.append(dumpb(k) + b" " + b + b";")

        return b"\n".join(entries)

    elif isinstance(data, FoamDict.DimensionSet) or (
        kind == Kind.DIMENSIONS and is_sequence(data) and len(data) == 7
    ):
        return b"[" + b" ".join(dumpb(v) for v in data) + b"]"

    elif (kind == Kind.FIELD or kind == Kind.BINARY_FIELD) and (
        isinstance(data, (int, float))
        or is_sequence(data)
        and data
        and isinstance(data[0], (int, float))
        and len(data) in (3, 6, 9)
    ):
        return b"uniform " + dumpb(data, kind=Kind.SINGLE_ENTRY)

    elif (kind == Kind.FIELD or kind == Kind.BINARY_FIELD) and is_sequence(data):
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

    elif kind != Kind.SINGLE_ENTRY and isinstance(data, tuple):
        return b" ".join(dumpb(v) for v in data)

    elif isinstance(data, FoamDict.Dimensioned):
        if data.name is not None:
            return (
                dumpb(data.name)
                + b" "
                + dumpb(data.dimensions, kind=Kind.DIMENSIONS)
                + b" "
                + dumpb(data.value, kind=Kind.SINGLE_ENTRY)
            )
        else:
            return (
                dumpb(data.dimensions, kind=Kind.DIMENSIONS)
                + b" "
                + dumpb(data.value, kind=Kind.SINGLE_ENTRY)
            )

    elif is_sequence(data):
        return b"(" + b" ".join(dumpb(v, kind=Kind.SINGLE_ENTRY) for v in data) + b")"

    elif data is True:
        return b"yes"
    elif data is False:
        return b"no"

    else:
        return str(data).encode("latin-1")
