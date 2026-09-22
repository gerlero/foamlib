from collections.abc import Iterable, MutableMapping
from typing import Protocol, overload

from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping


@overload
def add_to_mapping[K, V](
    d: MultiDict[K, V],
    key: K,
    value: V,
    /,
) -> MultiDict[K, V]: ...


@overload
def add_to_mapping[K, V](
    d: dict[K, V] | MultiDict[K, V],
    key: K,
    value: V,
    /,
) -> dict[K, V] | MultiDict[K, V]: ...


@overload
def add_to_mapping[K, V](
    d: MutableMultiMapping[K, V],
    key: K,
    value: V,
    /,
) -> MutableMultiMapping[K, V]: ...


@overload
def add_to_mapping[K, V](
    d: MutableMapping[K, V],
    key: K,
    value: V,
    /,
) -> MutableMapping[K, V]: ...


def add_to_mapping[K, V](
    d: MutableMapping[K, V],
    key: K,
    value: V,
    /,
) -> MutableMapping[K, V]:
    if isinstance(d, MutableMultiMapping):
        d.add(key, value)
        return d

    if key not in d:
        d[key] = value
        return d

    ret = MultiDict(d)
    ret.add(key, value)
    return ret


class SupportsKeysAndGetItem[K, V](Protocol):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...
