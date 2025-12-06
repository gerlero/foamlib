from collections.abc import Iterable, MutableMapping
from typing import Protocol, TypeVar, overload

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)


@overload
def add_to_mapping(
    d: MutableMultiMapping[_K, _V],
    key: _K,
    value: _V,
    /,
) -> MutableMultiMapping[_K, _V]: ...


@overload
def add_to_mapping(
    d: MutableMapping[_K, _V],
    key: _K,
    value: _V,
    /,
) -> MutableMapping[_K, _V]: ...


def add_to_mapping(
    d: MutableMapping[_K, _V],
    key: _K,
    value: _V,
    /,
) -> MutableMapping[_K, _V]:
    if isinstance(d, MutableMultiMapping):
        d.add(key, value)  # ty: ignore[invalid-argument-type]
        return d

    if key not in d:
        d[key] = value
        return d

    ret = MultiDict(d)
    ret.add(key, value)
    return ret


class SupportsKeysAndGetItem(Protocol[_K, _V_co]):
    def keys(self) -> Iterable[_K]: ...
    def __getitem__(self, key: _K) -> _V_co: ...
