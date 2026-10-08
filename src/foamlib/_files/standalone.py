import contextlib
import io
import sys
from collections.abc import Mapping

if sys.version_info >= (3, 14):
    from io import Reader, Writer
else:
    from typing_extensions import Reader, Writer

from multicollections import MultiDict

from .._files import _common, _serialization
from ..typing import (
    Data,
    FileDict,
    FileDictLike,
    StandaloneData,
    StandaloneDataLike,
    SubDict,
)
from ._normalization import normalized
from ._parsing import parse


def dump(
    value: FileDictLike | StandaloneDataLike,
    fp: Writer[bytes],
    /,
    *,
    ensure_header: bool = True,
) -> None:
    """
    Standalone serializing function (i.e., does not use :class:`foamlib.FoamFile`).

    Serialize Python objects directly to the OpenFOAM FoamFile format.

    For more convenience when working with OpenFOAM cases, it is recommended to use
    the :class:`foamlib.FoamFile` class instead of calling this function.

    :param value: The Python object to serialize. This can be a dictionary, list,
        or any other object that can be serialized to the OpenFOAM format.
    :param fp: A binary file-like object to write the serialized data to.
    :param ensure_header: Whether to include the "FoamFile" header in the output.
        If ``True``, a header will be included if it is not already present in the
        input object.
    """
    if not isinstance(value, Mapping):
        value = {None: value}

    value = normalized(value, target=FileDict)

    if "FoamFile" not in value and ensure_header:
        class_ = "dictionary"
        with contextlib.suppress(KeyError, TypeError):
            class_ = _common.vol_field_class(value["internalField"])

        new = MultiDict[str | None, StandaloneData | Data | SubDict | None](
            FoamFile={"version": 2.0, "format": "ascii", "class": class_}
        )
        new.extend(value)
        value = new

    _serialization.dump(value, fp)


def dumps(
    value: FileDictLike | StandaloneDataLike, /, *, ensure_header: bool = True
) -> bytes:
    """
    Standalone serializing function (i.e., does not use :class:`foamlib.FoamFile`).

    Serialize Python objects directly to the OpenFOAM FoamFile format.

    For more convenience when working with OpenFOAM cases, it is recommended to use
    the :class:`foamlib.FoamFile` class instead of calling this function.

    :param value: The Python object to serialize. This can be a dictionary, list,
        or any other object that can be serialized to the OpenFOAM format.
    :param ensure_header: Whether to include the "FoamFile" header in the output.
        If ``True``, a header will be included if it is not already present in the
        input object.
    """
    with io.BytesIO() as f:
        dump(value, f, ensure_header=ensure_header)
        return f.getvalue()


def load(
    fp: Reader[bytes | str], /, *, include_header: bool = False
) -> FileDict | StandaloneData:
    """
    Standalone deserializing function (i.e., does not use :class:`foamlib.FoamFile`).

    Deserialize the OpenFOAM FoamFile format directly to Python objects.

    For more convenience when working with OpenFOAM cases, it is recommended to use
    the :class:`foamlib.FoamFile` class instead of calling this function.

    :param fp: A file-like object to read from.
    :param include_header: Whether to include the "FoamFile" header in the output.
        If `True`, the header will be included if it is present in the input object.
    """
    return loads(fp.read(), include_header=include_header)


def loads(
    s: bytes | bytearray | str,
    /,
    *,
    include_header: bool = False,
) -> FileDict | StandaloneData:
    """
    Standalone deserializing function (i.e., does not use :class:`foamlib.FoamFile`).

    Deserialize the OpenFOAM FoamFile format directly to Python objects.

    For more convenience when working with OpenFOAM cases, it is recommended to use
    the :class:`foamlib.FoamFile` class instead of calling this function.

    :param s: The string to deserialize. This can be a dictionary, list, or any
        other object that can be serialized to the OpenFOAM format.
    :param include_header: Whether to include the "FoamFile" header in the output.
        If `True`, the header will be included if it is present in the input object.
    """
    file = parse(s, target=FileDict)

    if not include_header:
        file.pop("FoamFile", None)

    if len(file) == 1 and None in file:
        return file[None]  # ty: ignore[invalid-return-type]

    return file
