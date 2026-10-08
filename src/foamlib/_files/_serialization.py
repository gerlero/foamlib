import io
import sys
from collections.abc import Buffer, Mapping
from typing import Literal, assert_never

if sys.version_info >= (3, 14):
    from io import Writer
else:
    from typing_extensions import Writer

import numpy as np

from .._files import _common
from .types import _NAMED_DIMENSION_IDS, Dimensioned, DimensionSet
from .typing import (
    Data,
    Dict,
    FileDict,
    KeywordEntry,
    StandaloneData,
    SubDict,
)


class _PrefixedWriter[B: Buffer](Writer[B]):
    def __init__(self, writer: Writer, prefix: bytes, /) -> None:
        self._writer = writer
        self._prefix = prefix

    def write(self, b: B, /) -> int:
        if self._prefix and b:
            self._writer.write(self._prefix)
            self._prefix = b""
        return self._writer.write(b)


def dump(
    data: FileDict
    | Data
    | StandaloneData
    | KeywordEntry
    | SubDict
    | Dict
    | np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]],
    fp: Writer[bytes],
    /,
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
    _tuple_is_keyword_entry: bool = False,
) -> None:
    match data, keywords, format_:
        case {"FoamFile": {"format": ("ascii" | "binary") as format_}}, (), None:  # ty: ignore[invalid-assignment]
            pass
    match data, keywords, format_:
        case {}, _, _:
            if keywords != ():
                fp.write(b"{")
            first = True
            for k, v in data.items():
                if not first:
                    fp.write(b" ")
                first = False
                if k is not None:
                    dump(
                        (k, v),  # ty: ignore[invalid-argument-type]
                        fp,
                        keywords=keywords,
                        format_=format_,
                        _tuple_is_keyword_entry=True,
                    )
                else:
                    dump(
                        v,  # ty: ignore[invalid-argument-type]
                        fp,
                        keywords=keywords,
                        format_=format_,
                    )
            if keywords != ():
                fp.write(b"}")

        case float(), _common.FIELD_KEYWORDS, _:
            fp.write(b"uniform ")
            dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(3,) | (6,) | (9,)), _common.FIELD_KEYWORDS, _:
            fp.write(b"uniform ")
            dump(data.tolist(), fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]
        case np.ndarray(shape=(_,)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<scalar> ")
            dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(_, 3)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<vector> ")
            dump(data, fp, keywords=None, format_=format_)
        case np.ndarray(shape=(_, 6)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<symmTensor> ")
            dump(data, fp, keywords=None, format_=format_)

        case np.ndarray(shape=(_, 9)), _common.FIELD_KEYWORDS, _:
            fp.write(b"nonuniform List<tensor> ")
            dump(data, fp, keywords=None, format_=format_)
        case np.ndarray(), _, "binary":
            dump(len(data), fp, keywords=None, format_=None)
            fp.write(b"(")
            fp.write(data.tobytes())
            fp.write(b")")

        case np.ndarray(), (_, *_) | None, "ascii" | None:
            dump(len(data), fp, keywords=None, format_=None)
            dump(
                data.tolist(),  # ty: ignore[invalid-argument-type]
                fp,
                keywords=None,
                format_=format_,
            )
        case np.ndarray(), (), "ascii" | None:
            dump(data.tolist(), fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]

        case DimensionSet(), _, _:
            try:
                name = _NAMED_DIMENSION_IDS[id(data)]
            except KeyError:
                fp.write(b"[")
                dump(tuple(data), fp, keywords=None, format_=format_)
                fp.write(b"]")
            else:
                fp.write(b"[")
                dump(name, fp, keywords=None, format_=format_)
                fp.write(b"]")
        case Dimensioned(name=None), _, _:
            dump(data.dimensions, fp, keywords=None, format_=format_)
            fp.write(b" ")
            dump(data.value, fp, keywords=None, format_=format_)
        case Dimensioned(name=str()), _, _:
            dump(data.name, fp, keywords=None, format_=format_)  # ty: ignore[invalid-argument-type]
            fp.write(b" ")
            dump(data.dimensions, fp, keywords=None, format_=format_)
            fp.write(b" ")
            dump(data.value, fp, keywords=None, format_=format_)
        case (
            tuple((_, _)),
            _,
            _,
        ) if _tuple_is_keyword_entry and not isinstance(data, DimensionSet):
            assert len(data) == 2
            k, v = data
            if isinstance(k, str) and k[0] == "#":
                fp.write(b"\n")
            if k is not None:
                dump(k, fp, keywords=keywords)
            dump(
                v,
                _PrefixedWriter(fp, b" ") if k is not None else fp,
                keywords=(*keywords, k)  # ty: ignore[invalid-argument-type]
                if keywords is not None and k is not None
                else ()
                if k is None
                else None,
                format_=format_,
            )
            if isinstance(k, str) and k[0] == "#":
                fp.write(b"\n")
            elif (
                k is not None
                and not isinstance(v, Mapping)
                and not (isinstance(k, str) and k.startswith("$") and v is None)
            ):
                fp.write(b";")

        case tuple((_, _, *_)), _, _ if not isinstance(data, DimensionSet):
            for i, v in enumerate(data):
                if i:
                    fp.write(b" ")
                dump(v, fp, keywords=keywords, format_=format_)
        case [*_], _, _:
            fp.write(b"(")
            for i, v in enumerate(data):
                if i:
                    fp.write(b" ")
                dump(
                    v,  # ty: ignore[invalid-argument-type]
                    fp,
                    keywords=None,
                    format_=format_,
                    _tuple_is_keyword_entry=True,
                )
            fp.write(b")")

        case None, _, _:
            pass
        case True, _, _:
            fp.write(b"yes")

        case False, _, _:
            fp.write(b"no")

        case float() | int(), _, _:
            fp.write(str(data).encode())

        case str(), _, _:
            fp.write(data.encode())

        case _:
            assert_never(data)  # ty: ignore[type-assertion-failure]


def dumps(
    data: FileDict
    | Data
    | StandaloneData
    | KeywordEntry
    | SubDict
    | Dict
    | np.ndarray[tuple[Literal[3, 4]], np.dtype[np.int64]],
    /,
    *,
    keywords: tuple[str, ...] | None = (),
    format_: Literal["ascii", "binary"] | None = None,
    _tuple_is_keyword_entry: bool = False,
) -> bytes:
    with io.BytesIO() as f:
        dump(
            data,
            f,
            keywords=keywords,
            format_=format_,
            _tuple_is_keyword_entry=_tuple_is_keyword_entry,
        )
        return f.getvalue()
