from __future__ import annotations

import sys
from typing import Protocol, Sequence, TypeVar

import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Iterable, MutableMapping
else:
    from typing import Iterable, MutableMapping

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)
_MM = TypeVar("_MM", bound=MutableMapping)


def is_sequence(
    value: object,
) -> TypeGuard[Sequence[object] | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    return (isinstance(value, Sequence) and not isinstance(value, str)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    )


def add_to_mapping(
    d: _MM[_K, _V],
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


def as_dict_check_unique(items: Iterable[tuple[_K, _V]]) -> dict[_K, _V]:
    ret = {}
    for key, value in items:
        if key in ret:
            msg = f"Duplicate key found: {key}"
            raise ValueError(msg)
        ret[key] = value
    return ret


class SupportsKeysAndGetItem(Protocol[_K, _V_co]):
    def keys(self) -> Iterable[_K]: ...
    def __getitem__(self, key: _K) -> _V_co: ...
