from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING, overload

if sys.version_info >= (3, 9):
    from collections.abc import Mapping
else:
    from typing import Mapping

import numpy as np
from multicollections import MultiDict

from ._parsing import Parsed
from ._util import as_dict_check_unique, is_sequence
from .types import Dimensioned, DimensionSet

if TYPE_CHECKING:
    from ._typing import (
        Data,
        DataLike,
        KeywordEntryLike,
        StandaloneData,
        StandaloneDataLike,
        SubDict,
        SubDictLike,
    )


@overload
def normalize(
    data: DataLike, *, keywords: tuple[str, ...] | None = None, bool_ok: bool = True
) -> Data: ...


@overload
def normalize(
    data: StandaloneDataLike,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> StandaloneData: ...


@overload
def normalize(
    data: SubDictLike, *, keywords: tuple[str, ...] | None = None, bool_ok: bool = True
) -> SubDict: ...


def normalize(
    data: DataLike | StandaloneDataLike | SubDictLike,
    *,
    keywords: tuple[str, ...] | None = None,
    bool_ok: bool = True,
) -> Data | StandaloneData | SubDict:
    if isinstance(data, Mapping):
        items = ((normalize(k, bool_ok=False), normalize(v)) for k, v in data.items())
        if keywords is None:
            return as_dict_check_unique(items)
        ret1: SubDict = MultiDict(items)
        seen = set()
        for k, v in ret1.items():
            if k.startswith("#"):
                if isinstance(v, Mapping):
                    msg = f"Directive {k} cannot have a dictionary as value"
                    raise ValueError(msg)
            else:
                if k in seen:
                    msg = (
                        f"Duplicate keyword {k} in dictionary with keywords {keywords}"
                    )
                    raise ValueError(msg)
                seen.add(k)
        return ret1

    if keywords == () and is_sequence(data) and not isinstance(data, tuple):
        try:
            arr = np.asarray(data)
        except ValueError:
            pass
        else:
            if np.issubdtype(arr.dtype, np.integer) and arr.ndim == 1:
                return arr
            if arr.ndim == 2 and arr.shape[1] == 3:
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                return arr

    if keywords is not None and (
        keywords == ("internalField",)
        or (
            len(keywords) == 3
            and keywords[0] == "boundaryField"
            and (
                keywords[2] == "value"
                or keywords[2] == "gradient"
                or keywords[2].endswith(("Value", "Gradient"))
            )
        )
    ):
        if is_sequence(data) and not isinstance(data, tuple):
            try:
                arr = np.asarray(data)
            except ValueError:
                pass
            else:
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)

                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] in (3, 6, 9)):
                    return arr

            return [normalize(d) for d in data]

        if isinstance(data, int):
            return float(data)

        return normalize(data)

    if isinstance(data, np.ndarray):
        ret2 = data.tolist()
        assert isinstance(ret2, (int, float, list))
        return ret2

    if (
        not isinstance(data, DimensionSet)
        and keywords is not None
        and keywords == ("dimensions",)
        and is_sequence(data)
        and len(data) <= 7
        and all(isinstance(d, (int, float)) for d in data)
    ):
        return DimensionSet(*data)

    if keywords is None and isinstance(data, tuple) and len(data) == 2:
        k2, v2 = data
        if isinstance(k2, Mapping):
            msg = "Keyword in keyword entry cannot be a dictionary"
            raise ValueError(msg)
        k2 = normalize(k2, bool_ok=False)
        v2 = normalize(v2)
        return (k2, v2)

    if (
        is_sequence(data)
        and not isinstance(data, DimensionSet)
        and not isinstance(data, tuple)
    ):
        return [normalize(d) for d in data]

    if isinstance(data, tuple) and not isinstance(data, DimensionSet):
        return tuple(normalize(d, keywords=keywords) for d in data)

    if isinstance(data, str):
        with contextlib.suppress(ValueError, KeyError):
            s = Parsed(data)[()]
            if not bool_ok and isinstance(s, bool):
                return data
            if isinstance(s, (str, tuple, bool)):
                return s

    if isinstance(
        data,
        (int, float, bool, DimensionSet, Dimensioned),
    ):
        return data

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def dumps(
    data: DataLike | StandaloneDataLike | KeywordEntryLike | SubDictLike,
    *,
    keywords: tuple[str, ...] | None = None,
    header: SubDictLike | None = None,
    tuple_is_keyword_entry: bool = False,
) -> bytes:
    data = normalize(data, keywords=keywords)

    if isinstance(data, Mapping):
        return (
            b"{"
            + b" ".join(
                dumps(
                    (k, v),
                    keywords=keywords,
                    tuple_is_keyword_entry=True,
                )
                for k, v in data.items()
            )
            + b"}"
        )

    if keywords == () and isinstance(data, np.ndarray):
        if (header.get("format", "") if header else "") == "binary":
            return dumps(len(data)) + b"(" + data.tobytes() + b")"

        return dumps(data.tolist())

    if (
        keywords is not None
        and (
            keywords == ("internalField",)
            or (
                len(keywords) == 3
                and keywords[0] == "boundaryField"
                and (
                    keywords[2] == "value"
                    or keywords[2] == "gradient"
                    or keywords[2].endswith(("Value", "Gradient"))
                )
            )
        )
        and isinstance(data, (int, float, np.ndarray))
    ):
        data = np.asarray(data)
        class_ = header.get("class", "") if header else ""
        assert isinstance(class_, str)
        scalar = "Scalar" in class_

        shape = np.shape(data)
        if not shape or (not scalar and shape in ((3,), (6,), (9,))):
            return b"uniform " + dumps(data)

        assert isinstance(data, np.ndarray)
        ndim = np.ndim(data)
        if ndim == 1:
            tensor_kind = b"scalar"

        elif ndim == 2:
            assert len(shape) == 2
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

        binary = (header.get("format", "") if header else "") == "binary"

        contents = b"(" + data.tobytes() + b")" if binary else dumps(data)

        return b"nonuniform List<" + tensor_kind + b"> " + dumps(len(data)) + contents

    if isinstance(data, DimensionSet):
        return b"[" + b" ".join(dumps(v) for v in data) + b"]"

    if isinstance(data, Dimensioned):
        if data.name is not None:
            return (
                dumps(data.name)
                + b" "
                + dumps(data.dimensions)
                + b" "
                + dumps(data.value)
            )
        return dumps(data.dimensions) + b" " + dumps(data.value)

    if isinstance(data, tuple):
        if tuple_is_keyword_entry:
            k, v = data
            ret = b"\n" if isinstance(k, str) and k[0] == "#" else b""
            ret += dumps(k)
            val = dumps(
                v,
                keywords=(*keywords, k)
                if keywords is not None and isinstance(k, str)
                else None,
            )
            if val:
                ret += b" " + val
            if isinstance(k, str) and k[0] == "#":
                ret += b"\n"
            elif not isinstance(v, Mapping):
                ret += b";"
            return ret

        return b" ".join(dumps(v, keywords=keywords, header=header) for v in data)

    if is_sequence(data):
        return (
            b"(" + b" ".join(dumps(v, tuple_is_keyword_entry=True) for v in data) + b")"
        )

    if data is True:
        return b"yes"
    if data is False:
        return b"no"

    return str(data).encode("latin-1")
