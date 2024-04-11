import sys
from contextlib import suppress

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

from .._util import is_sequence
from ._base import FoamDictionaryBase


def _serialize_switch(value: FoamDictionaryBase._SetValue) -> str:
    if value is True:
        return "yes"
    elif value is False:
        return "no"
    else:
        raise TypeError(f"Not a bool: {type(value)}")


def _serialize_list(
    value: FoamDictionaryBase._SetValue,
) -> str:
    if is_sequence(value):
        return f"({' '.join(_serialize_value(v) for v in value)})"
    else:
        raise TypeError(f"Not a valid sequence: {type(value)}")


def _serialize_field(
    value: FoamDictionaryBase._SetValue,
) -> str:
    if is_sequence(value):
        try:
            s = _serialize_list(value)
        except TypeError:
            raise TypeError(f"Not a valid field: {type(value)}") from None
        else:
            if not is_sequence(value[0]) and len(value) < 10:
                return f"uniform {s}"
            else:
                if not is_sequence(value[0]):
                    kind = "scalar"
                elif len(value[0]) == 3:
                    kind = "vector"
                elif len(value[0]) == 6:
                    kind = "symmTensor"
                elif len(value[0]) == 9:
                    kind = "tensor"
                else:
                    raise TypeError(
                        f"Unsupported sequence length for field: {len(value[0])}"
                    )
                return f"nonuniform List<{kind}> {len(value)}{s}"
    else:
        return f"uniform {value}"


def _serialize_dimensions(
    value: FoamDictionaryBase._SetValue,
) -> str:
    if is_sequence(value) and len(value) == 7:
        return f"[{' '.join(str(v) for v in value)}]"
    else:
        raise TypeError(f"Not a valid dimension set: {type(value)}")


def _serialize_dimensioned(
    value: FoamDictionaryBase._SetValue,
) -> str:
    if isinstance(value, FoamDictionaryBase.Dimensioned):
        if value.name is not None:
            return f"{value.name} {_serialize_dimensions(value.dimensions)} {_serialize_value(value.value)}"
        else:
            return f"{_serialize_dimensions(value.dimensions)} {_serialize_value(value.value)}"
    else:
        raise TypeError(f"Not a valid dimensioned value: {type(value)}")


def _serialize_value(
    value: FoamDictionaryBase._SetValue,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
) -> str:
    if isinstance(value, FoamDictionaryBase.DimensionSet) or assume_dimensions:
        with suppress(TypeError):
            return _serialize_dimensions(value)

    if assume_field:
        with suppress(TypeError):
            return _serialize_field(value)

    with suppress(TypeError):
        return _serialize_dimensioned(value)

    with suppress(TypeError):
        return _serialize_list(value)

    with suppress(TypeError):
        return _serialize_switch(value)

    with suppress(TypeError):
        return _serialize_dictionary(value)

    return str(value)


def _serialize_dictionary(
    value: FoamDictionaryBase._SetValue,
) -> str:
    if isinstance(value, Mapping):
        return "\n".join(serialize_entry(k, v) for k, v in value.items())
    else:
        raise TypeError(f"Not a valid dictionary: {type(value)}")


def serialize_entry(
    keyword: str,
    value: FoamDictionaryBase._SetValue,
    *,
    assume_field: bool = False,
    assume_dimensions: bool = False,
) -> str:
    try:
        return f"{keyword}\n{{\n{_serialize_dictionary(value)}\n}}"
    except TypeError:
        return f"{keyword} {_serialize_value(value, assume_field=assume_field, assume_dimensions=assume_dimensions)};"
