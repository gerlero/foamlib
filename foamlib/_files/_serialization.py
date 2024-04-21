import array
import itertools
import sys

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


def dumpb(
    data: FoamDict._SetData,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
    assume_data_entries: bool = False,
    binary_fields: bool = False,
) -> bytes:
    if numpy and isinstance(data, np.ndarray):
        return dumpb(
            data.tolist(),
            assume_field=assume_field,
            assume_dimensions=assume_dimensions,
            assume_data_entries=assume_data_entries,
            binary_fields=binary_fields,
        )

    elif isinstance(data, Mapping):
        entries = []
        for k, v in data.items():
            b = dumpb(
                v,
                assume_field=assume_field,
                assume_dimensions=assume_dimensions,
                assume_data_entries=True,
                binary_fields=binary_fields,
            )
            if isinstance(v, Mapping):
                entries.append(dumpb(k) + b"\n" + b"{\n" + b + b"\n}")
            elif b:
                entries.append(dumpb(k) + b" " + b + b";")
            else:
                entries.append(dumpb(k) + b";")

        return b"\n".join(entries)

    elif isinstance(data, FoamDict.DimensionSet) or (
        assume_dimensions and is_sequence(data) and len(data) == 7
    ):
        return b"[" + b" ".join(dumpb(v) for v in data) + b"]"

    elif assume_field and (
        isinstance(data, (int, float))
        or is_sequence(data)
        and data
        and isinstance(data[0], (int, float))
        and len(data) in (3, 6, 9)
    ):
        return b"uniform " + dumpb(data)

    elif assume_field and is_sequence(data):
        if isinstance(data[0], (int, float)):
            kind = b"scalar"
        elif len(data[0]) == 3:
            kind = b"vector"
        elif len(data[0]) == 6:
            kind = b"symmTensor"
        elif len(data[0]) == 9:
            kind = b"tensor"
        else:
            return dumpb(
                data,
                assume_dimensions=assume_dimensions,
                assume_data_entries=assume_data_entries,
                binary_fields=binary_fields,
            )

        if binary_fields:
            if kind == b"scalar":
                contents = b"(" + array.array("d", data).tobytes() + b")"
            else:
                contents = (
                    b"("
                    + array.array("d", itertools.chain.from_iterable(data)).tobytes()
                    + b")"
                )
        else:
            contents = dumpb(data)

        return b"nonuniform List<" + kind + b"> " + dumpb(len(data)) + contents

    elif assume_data_entries and isinstance(data, tuple):
        return b" ".join(
            dumpb(
                v,
                assume_field=assume_field,
                assume_dimensions=assume_dimensions,
                binary_fields=binary_fields,
            )
            for v in data
        )

    elif isinstance(data, FoamDict.Dimensioned):
        if data.name is not None:
            return (
                dumpb(data.name)
                + b" "
                + dumpb(data.dimensions, assume_dimensions=True)
                + b" "
                + dumpb(data.value)
            )
        else:
            return (
                dumpb(data.dimensions, assume_dimensions=True)
                + b" "
                + dumpb(data.value)
            )

    elif is_sequence(data):
        return b"(" + b" ".join(dumpb(v) for v in data) + b")"

    elif data is True:
        return b"yes"
    elif data is False:
        return b"no"

    else:
        return str(data).encode("latin-1")
