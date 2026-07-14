from collections.abc import Mapping
from typing import Literal, assert_never

import numpy as np

from .._files import _common
from ..typing import (
    Data,
    Dict,
    FileDict,
    KeywordEntry,
    StandaloneData,
    SubDict,
)
from .types import Dimensioned, DimensionSet


def dumps(
    data: FileDict
    | Data
    | StandaloneData
    | KeywordEntry
    | SubDict
    | Dict
    | np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]],
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
    _tuple_is_keyword_entry: bool = False,
) -> bytes:
    match data, keywords, format_:
        case {"FoamFile": {"format": ("ascii" | "binary") as format_}}, (), None:  # ty: ignore[invalid-assignment]
            pass

    match data, keywords, format_:
        case {}, _, _:
            return (
                (b"{" if keywords != () else b"")
                + b" ".join(
                    dumps(
                        (k, v),
                        keywords=keywords,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                    if k is not None
                    else dumps(
                        v,  # ty: ignore[invalid-argument-type]
                        keywords=keywords,
                        format_=format_,
                    )
                    for k, v in data.items()
                )
                + (b"}" if keywords != () else b"")
            )

        case float(), _common.FIELD_KEYWORDS, _:
            return b"uniform " + dumps(data, keywords=None, format_=format_)

        case np.ndarray(shape=(3,) | (6,) | (9,)), _common.FIELD_KEYWORDS, _:
            return b"uniform " + dumps(data.tolist(), keywords=None, format_=format_)  # ty: ignore[no-matching-overload]

        case np.ndarray(shape=(_,)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<scalar> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 3)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<vector> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 6)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<symmTensor> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(shape=(_, 9)), _common.FIELD_KEYWORDS, _:
            return b"nonuniform List<tensor> " + dumps(
                data, keywords=None, format_=format_
            )

        case np.ndarray(), _, "binary":
            return (
                dumps(len(data), keywords=None, format_=None)
                + b"("
                + data.tobytes()
                + b")"
            )

        case np.ndarray(), (_, *_) | None, "ascii" | None:
            return dumps(len(data), keywords=None, format_=None) + dumps(
                data.tolist(),  # ty: ignore[no-matching-overload]
                keywords=None,
                format_=format_,
            )

        case np.ndarray(), (), "ascii" | None:
            return dumps(data.tolist(), keywords=None, format_=format_)  # ty: ignore[no-matching-overload]

        case DimensionSet(), _, _:
            return b"[" + dumps(tuple(data), keywords=None, format_=format_) + b"]"

        case Dimensioned(name=None), _, _:
            return (
                dumps(data.dimensions, keywords=None, format_=format_)
                + b" "
                + dumps(data.value, keywords=None, format_=format_)
            )

        case Dimensioned(name=str()), _, _:
            return (
                dumps(data.name, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]
                + b" "
                + dumps(data.dimensions, keywords=None, format_=format_)
                + b" "
                + dumps(data.value, keywords=None, format_=format_)
            )

        case (
            tuple((_, _, *_)),
            _,
            _,
        ) if not isinstance(data, DimensionSet):
            if _tuple_is_keyword_entry:
                k, v = data

                ret = b"\n" if isinstance(k, str) and k[0] == "#" else b""
                if k is not None:
                    ret += dumps(k, keywords=keywords)  # ty: ignore[invalid-argument-type]
                val = dumps(
                    v,  # ty: ignore[invalid-argument-type]
                    keywords=(*keywords, k)  # ty: ignore[invalid-argument-type]
                    if keywords is not None and k is not None
                    else ()
                    if k is None
                    else None,
                    format_=format_,
                )
                if k is not None and val:
                    ret += b" "
                ret += val
                if isinstance(k, str) and k[0] == "#":
                    ret += b"\n"
                elif k is not None and not isinstance(v, Mapping):
                    ret += b";"
                return ret

            return b" ".join(dumps(v, keywords=keywords, format_=format_) for v in data)  # ty: ignore[invalid-argument-type]

        case [*_], _, _:
            return (
                b"("
                + b" ".join(
                    dumps(
                        v,  # ty: ignore[invalid-argument-type]
                        keywords=None,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                    for v in data
                )
                + b")"
            )

        case None, _, _:
            return b""

        case True, _, _:
            return b"yes"

        case False, _, _:
            return b"no"

        case float() | int(), _, _:
            return dumps(str(data))

        case str(), _, _:
            return data.encode()

        case _:
            assert_never(data)  # ty: ignore[type-assertion-failure]
