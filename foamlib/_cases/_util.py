import functools
import sys
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
)

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator


R = TypeVar("R")


class _AwaitableAsyncContextManager(Generic[R]):
    def __init__(self, cm: "AsyncContextManager[R]"):
        self._cm = cm

    def __await__(self) -> Generator[Any, Any, R]:
        return self._cm.__aenter__().__await__()

    async def __aenter__(self) -> R:
        return await self._cm.__aenter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        return await self._cm.__aexit__(exc_type, exc_val, exc_tb)


def awaitableasynccontextmanager(
    cm: Callable[..., "AsyncContextManager[R]"],
) -> Callable[..., _AwaitableAsyncContextManager[R]]:
    @functools.wraps(cm)
    def f(*args: Any, **kwargs: Any) -> _AwaitableAsyncContextManager[R]:
        return _AwaitableAsyncContextManager(cm(*args, **kwargs))

    return f
