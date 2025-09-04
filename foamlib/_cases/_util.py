from __future__ import annotations

import functools
import sys
import threading
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
    from contextlib import AbstractAsyncContextManager, AbstractContextManager
else:
    from typing import AsyncContextManager as AbstractAsyncContextManager
    from typing import AsyncGenerator, Awaitable, Callable, Generator
    from typing import ContextManager as AbstractContextManager

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from types import TracebackType


Y = TypeVar("Y")
S = TypeVar("S")
R = TypeVar("R")


class ValuedGenerator(Generic[Y, S, R]):
    def __init__(self, generator: Generator[Y, S, R], /) -> None:
        self._generator = generator

    def __iter__(self) -> Generator[Y, S, R]:
        self.value = yield from self._generator
        return self.value


class AwaitableAsyncContextManager(AbstractAsyncContextManager[R], Awaitable[R]):
    def __init__(self, cm: AbstractAsyncContextManager[R], /) -> None:
        self._cm = cm

    @override
    def __await__(self) -> Generator[Any, Any, R]:
        return self._cm.__aenter__().__await__()

    @override
    async def __aenter__(self) -> R:
        return await self._cm.__aenter__()

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._cm.__aexit__(exc_type, exc_val, exc_tb)


def awaitableasynccontextmanager(
    func: Callable[..., AsyncGenerator[R]], /
) -> Callable[..., AwaitableAsyncContextManager[R]]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> AwaitableAsyncContextManager[R]:
        return AwaitableAsyncContextManager(asynccontextmanager(func)(*args, **kwargs))

    return wrapper


class SingletonContextManager(AbstractContextManager[R]):
    def __init__(self, factory: Callable[[], AbstractContextManager[R]], /) -> None:
        self._factory = factory
        self._users = 0
        self._cm: AbstractContextManager[R] | None = None
        self._ret: R | None = None
        self._lock = threading.Lock()

    @override
    def __enter__(self) -> R:
        with self._lock:
            if self._users == 0:
                self._cm = self._factory()
                self._ret = self._cm.__enter__()
            self._users += 1
            return cast("R", self._ret)

    @override
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
