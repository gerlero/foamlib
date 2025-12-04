from collections.abc import Iterable, MutableMapping, Sequence
from typing import Literal, Protocol, TypeGuard, TypeVar, overload

import numpy as np
from multicollections import MultiDict
from multicollections.abc import MutableMultiMapping

_K = TypeVar("_K")
_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)


def is_sequence(
    value: object,
    /,
) -> TypeGuard[Sequence[object] | np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    return (isinstance(value, Sequence) and not isinstance(value, str)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    )


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


@overload
def as_dict(
    items: Iterable[tuple[_K, _V]],
    *,
    multi_ok: Literal[False] = ...,
) -> dict[_K, _V]: ...


@overload
def as_dict(
    items: Iterable[tuple[_K, _V]],
    *,
    multi_ok: Literal[True] = ...,
) -> dict[_K, _V] | MultiDict[_K, _V]: ...


def as_dict(
    items: Iterable[tuple[_K, _V]], *, multi_ok: bool = False
) -> dict[_K, _V] | MultiDict[_K, _V]:
    if multi_ok:
        ret: dict[_K, _V] | MultiDict[_K, _V] = {}
        for key, value in items:
            ret = add_to_mapping(ret, key, value)  # ty: ignore[invalid-assignment]
        return ret

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
