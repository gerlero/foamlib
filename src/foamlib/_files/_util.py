from collections.abc import Iterable, MutableMapping
from typing import Protocol, TypeVar, overload

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)
_M = TypeVar("_M", bound=MutableMapping[_K, _V])  # ty: ignore[invalid-type-variable-bound]
_MM = TypeVar("_MM", bound=MutableMultiMapping[_K, _V])  # ty: ignore[invalid-type-variable-bound]


@overload
def add_to_mapping(
    d: _MM,
    key: _K,
    value: _V,
    /,
) -> _MM: ...


@overload
def add_to_mapping(
    d: _M,
    key: _K,
    value: _V,
    /,
) -> _M | MultiDict[_K, _V]: ...


def add_to_mapping(
    d: _M,
    key: _K,
    value: _V,
    /,
) -> _M | MultiDict[_K, _V]:
    if isinstance(d, MutableMultiMapping):
        d.add(key, value)  # ty: ignore[invalid-argument-type]
        return d

    if key not in d:  # ty: ignore[unsupported-operator]
        d[key] = value  # ty: ignore[invalid-assignment]
        return d

    ret = MultiDict(d)  # ty: ignore[invalid-argument-type]
    ret.add(key, value)
    return ret  # ty: ignore[invalid-return-type]


class SupportsKeysAndGetItem(Protocol[_K, _V_co]):
    def keys(self) -> Iterable[_K]: ...
    def __getitem__(self, key: _K) -> _V_co: ...
