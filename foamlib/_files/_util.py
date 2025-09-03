from __future__ import annotations

import sys
from typing import TypeVar

if sys.version_info >= (3, 9):
    from collections.abc import MutableMapping
else:
    from typing import MutableMapping

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")
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
