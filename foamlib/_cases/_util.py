from __future__ import annotations

import functools
import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable, Generator
    from types import TracebackType


Y = TypeVar("Y")
S = TypeVar("S")
R = TypeVar("R")


class ValuedGenerator(Generic[Y, S, R]):
    def __init__(self, generator: Generator[Y, S, R]) -> None:
        self._generator = generator

    def __iter__(self) -> Generator[Y, S, R]:
        self.value = yield from self._generator
        return self.value


class _AwaitableAsyncContextManager(Generic[R]):
    def __init__(self, cm: contextlib.AbstractAsyncContextManager[R]) -> None:
        self._cm = cm

    def __await__(self) -> Generator[Any, Any, R]:
        return self._cm.__aenter__().__await__()

    async def __aenter__(self) -> R:
        return await self._cm.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._cm.__aexit__(exc_type, exc_val, exc_tb)


def awaitableasynccontextmanager(
    cm: Callable[..., contextlib.AbstractAsyncContextManager[R]],
) -> Callable[..., _AwaitableAsyncContextManager[R]]:
    @functools.wraps(cm)
    def f(*args: Any, **kwargs: Any) -> _AwaitableAsyncContextManager[R]:
        return _AwaitableAsyncContextManager(cm(*args, **kwargs))

    return f


class SingletonContextManager(Generic[R]):
    def __init__(
        self, factory: Callable[[], contextlib.AbstractContextManager[R]]
    ) -> None:
        self._factory = factory
        self._users = 0
        self._cm: contextlib.AbstractContextManager[R] | None = None
        self._ret: R | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> R:
        with self._lock:
            if self._users == 0:
                self._cm = self._factory()
                self._ret = self._cm.__enter__()
            self._users += 1
            return cast("R", self._ret)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        with self._lock:
            self._users -= 1
            if self._users == 0:
                assert self._cm is not None
                return self._cm.__exit__(exc_type, exc_val, exc_tb)
            return False
