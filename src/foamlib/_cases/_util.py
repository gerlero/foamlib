from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, Any, AsyncContextManager, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from types import TracebackType

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator


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
    def __init__(self, cm: AsyncContextManager[R]) -> None:
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
    cm: Callable[..., AsyncContextManager[R]],
) -> Callable[..., _AwaitableAsyncContextManager[R]]:
    @functools.wraps(cm)
    def f(*args: Any, **kwargs: Any) -> _AwaitableAsyncContextManager[R]:
        return _AwaitableAsyncContextManager(cm(*args, **kwargs))

    return f
