from __future__ import annotations

import sys
from abc import abstractmethod
from typing import TypeVar, overload

if sys.version_info >= (3, 9):
    from collections.abc import Iterable, Mapping, MutableMapping
else:
    from typing import Iterable, Mapping, MutableMapping

K = TypeVar("K")
V = TypeVar("V")
D = TypeVar("D")


class MultiMapping(Mapping[K, V]):
    class _NoneType:
        pass

    _NONE = _NoneType()

    def __getitem__(self, key: K) -> V:
        return self.getone(key)

    @overload
    def getone(self, key: K) -> V: ...

    @overload
    def getone(self, key: K, default: D) -> V | D: ...

    def getone(self, key: K, default: D | _NoneType = _NONE) -> V | D:
        values = self._getall(key)
        try:
            return values[0]
        except IndexError:
            if default is not self._NONE:
                assert not isinstance(default, self._NoneType)
                return default
            raise KeyError(key) from None

    @overload
    def getall(self, key: K) -> list[V]: ...

    @overload
    def getall(self, key: K, default: D) -> list[V] | D: ...

    def getall(self, key: K, default: D | _NoneType = _NONE) -> list[V] | D:
        ret = self._getall(key)
        if not ret:
            if default is not self._NONE:
                assert not isinstance(default, self._NoneType)
                return default
            raise KeyError(key)
        return ret

    @abstractmethod
    def _getall(self, key: K) -> list[V]:
        raise NotImplementedError


class MutableMultiMapping(MutableMapping[K, V], MultiMapping[K, V]):
    @abstractmethod
    def add(self, key: K, value: V) -> None:
        raise NotImplementedError

    @overload
    def popall(self, key: K) -> list[V]: ...

    @overload
    def popall(self, key: K, default: D) -> list[V] | D: ...

    def popall(
        self, key: K, default: D | MultiMapping._NoneType = MultiMapping._NONE
    ) -> list[V] | D:
        ret = []
        try:
            while True:
                ret.append(self.pop(key))
        except KeyError:
            if not ret:
                if default is not self._NONE:
                    assert not isinstance(default, self._NoneType)
                    return default
                raise
            return ret

    def extend(
        self, other: Mapping[K, V] | Iterable[tuple[K, V]] = (), **kwargs: V
    ) -> None:
        if isinstance(other, Mapping):
            for k, v in other.items():
                self.add(k, v)
        else:
            for k, v in other:
                self.add(k, v)
        for k, v in kwargs.items():
            self.add(k, v)  # type: ignore [arg-type]

    def merge(
        self, other: Mapping[K, V] | Iterable[tuple[K, V]] = (), **kwargs: V
    ) -> None:
        if isinstance(other, Mapping):
            for k, v in other.items():
                if k not in self:
                    self.add(k, v)
        else:
            for k, v in other:
                if k not in self:
                    self.add(k, v)

        for k, v in kwargs.items():
            if k not in self:
                self.add(k, v)  # type: ignore [arg-type]
