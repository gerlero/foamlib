import sys
from contextlib import suppress

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

from .._util import is_sequence
from ._base import FoamDict


def _serialize_switch(data: FoamDict._SetData) -> str:
    if data is True:
        return "yes"
    elif data is False:
        return "no"
    else:
        raise TypeError(f"Not a bool: {type(data)}")


def _serialize_list(
    data: FoamDict._SetData,
) -> str:
    if is_sequence(data):
        return f"({' '.join(_serialize_data_entry(v) for v in data)})"
    else:
        raise TypeError(f"Not a valid sequence: {type(data)}")


def _serialize_field(
    data: FoamDict._SetData,
) -> str:
    if is_sequence(data):
        try:
            s = _serialize_list(data)
        except TypeError:
            raise TypeError(f"Not a valid field: {type(data)}") from None
        else:
            if not is_sequence(data[0]) and len(data) < 10:
                return f"uniform {s}"
            else:
                if not is_sequence(data[0]):
                    kind = "scalar"
                elif len(data[0]) == 3:
                    kind = "vector"
                elif len(data[0]) == 6:
                    kind = "symmTensor"
                elif len(data[0]) == 9:
                    kind = "tensor"
                else:
                    raise TypeError(
                        f"Unsupported sequence length for field: {len(data[0])}"
                    )
                return f"nonuniform List<{kind}> {len(data)}{s}"
    else:
        return f"uniform {data}"


def _serialize_dimensions(
    data: FoamDict._SetData,
) -> str:
    if is_sequence(data) and len(data) == 7:
        return f"[{' '.join(str(v) for v in data)}]"
    else:
        raise TypeError(f"Not a valid dimension set: {type(data)}")


def _serialize_dimensioned(
    data: FoamDict._SetData,
) -> str:
    if isinstance(data, FoamDict.Dimensioned):
        if data.name is not None:
            return f"{data.name} {_serialize_dimensions(data.dimensions)} {_serialize_data_entry(data.value)}"
        else:
            return f"{_serialize_dimensions(data.dimensions)} {_serialize_data_entry(data.value)}"
    else:
        raise TypeError(f"Not a valid dimensioned value: {type(data)}")


def _serialize_data_entry(
    data: FoamDict._SetData,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
) -> str:
    if isinstance(data, FoamDict.DimensionSet) or assume_dimensions:
        with suppress(TypeError):
            return _serialize_dimensions(data)

    if assume_field:
        with suppress(TypeError):
            return _serialize_field(data)

    with suppress(TypeError):
        return _serialize_dimensioned(data)

    with suppress(TypeError):
        return _serialize_list(data)

    with suppress(TypeError):
        return _serialize_switch(data)

    with suppress(TypeError):
        return _serialize_dictionary(data)

    return str(data)


def _serialize_data_entries(
    data: FoamDict._SetData,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
) -> str:
    if isinstance(data, FoamDict.DimensionSet) or assume_dimensions:
        with suppress(TypeError):
            return _serialize_dimensions(data)

    if assume_field:
        with suppress(TypeError):
            return _serialize_field(data)

    if isinstance(data, tuple):
        return " ".join(_serialize_data_entry(v) for v in data)

    return _serialize_data_entry(data)


def _serialize_dictionary(
    data: FoamDict._SetData,
) -> str:
    if isinstance(data, Mapping):
        return "\n".join(serialize_keyword_entry(k, v) for k, v in data.items())
    else:
        raise TypeError(f"Not a valid dictionary: {type(data)}")


def serialize_keyword_entry(
    keyword: str,
    data: FoamDict._SetData,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
) -> str:
    with suppress(TypeError):
        return f"{keyword}\n{{\n{_serialize_dictionary(data)}\n}}"

    data = _serialize_data_entries(
        data, assume_field=assume_field, assume_dimensions=assume_dimensions
    )

    if not data:
        return f"{keyword};"
    else:
        return f"{keyword} {data};"
