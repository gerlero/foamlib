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


def dumps(
    data: FoamDict._SetData,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
    assume_data_entries: bool = False,
) -> str:
    if numpy and isinstance(data, np.ndarray):
        return dumps(
            data.tolist(),
            assume_field=assume_field,
            assume_dimensions=assume_dimensions,
        )

    elif isinstance(data, Mapping):
        entries = []
        for k, v in data.items():
            s = dumps(
                v,
                assume_field=assume_field,
                assume_dimensions=assume_dimensions,
                assume_data_entries=True,
            )
            if isinstance(v, Mapping):
                entries.append(f"{k}\n{{\n{s}\n}}")
            elif s:
                entries.append(f"{k} {s};")
            else:
                entries.append(f"{k};")
        return "\n".join(entries)

    elif isinstance(data, FoamDict.DimensionSet) or (
        assume_dimensions and is_sequence(data) and len(data) == 7
    ):
        return f"[{' '.join(str(v) for v in data)}]"

    elif assume_field and isinstance(data, (int, float)):
        return f"uniform {data}"

    elif assume_field and is_sequence(data):
        if isinstance(data[0], (int, float)) and len(data) in (3, 6, 9):
            return f"uniform {dumps(data)}"
        elif isinstance(data[0], (int, float)):
            return f"nonuniform List<scalar> {len(data)}{dumps(data)}"
        elif len(data[0]) == 3:
            return f"nonuniform List<vector> {len(data)}{dumps(data)}"
        elif len(data[0]) == 6:
            return f"nonuniform List<symmTensor> {len(data)}{dumps(data)}"
        elif len(data[0]) == 9:
            return f"nonuniform List<tensor> {len(data)}{dumps(data)}"
        else:
            return dumps(
                data,
                assume_dimensions=assume_dimensions,
                assume_data_entries=assume_data_entries,
            )

    elif assume_data_entries and isinstance(data, tuple):
        return " ".join(
            dumps(v, assume_field=assume_field, assume_dimensions=assume_dimensions)
            for v in data
        )

    elif isinstance(data, FoamDict.Dimensioned):
        if data.name is not None:
            return f"{data.name} {dumps(data.dimensions, assume_dimensions=True)} {dumps(data.value)}"
        else:
            return (
                f"{dumps(data.dimensions, assume_dimensions=True)} {dumps(data.value)}"
            )

    elif is_sequence(data):
        return f"({' '.join(dumps(v) for v in data)})"

    elif data is True:
        return "yes"
    elif data is False:
        return "no"

    else:
        return str(data)
