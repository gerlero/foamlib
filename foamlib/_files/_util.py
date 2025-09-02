from __future__ import annotations

import sys
from typing import TypeVar

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, MutableMapping, Sequence
else:
    from typing import Mapping, MutableMapping, Sequence

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")


def as_any_dict(
    seq: Sequence[tuple[_K, _V]] | Mapping[_K, _V],
    /,
) -> dict[_K, _V] | MultiDict[_K, _V]:
    if len(d := dict(seq)) == len(seq):
        return d
    return MultiDict(seq)


_MM = TypeVar("_MM", bound=MutableMapping[_K, _V])  # type: ignore[valid-type]


def add_to_mapping(
    d: _MM,
    key: _K,
    value: _V,
    /,
) -> _MM | MultiDict[_K, _V]:
    if isinstance(d, MutableMultiMapping):
        d.add(key, value)
        return d

    if key not in d:
        d[key] = value
        return d

    ret: MultiDict[_K, _V] = MultiDict(d)
    ret.add(key, value)
    return ret
