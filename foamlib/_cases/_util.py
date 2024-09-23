import functools
from types import TracebackType
from typing import Any, AsyncContextManager, Callable, Optional, Type


class _AwaitableAsyncContextManager:
    def __init__(self, cm: "AsyncContextManager[Any]"):
        self._cm = cm

    def __await__(self) -> Any:
        return self._cm.__aenter__().__await__()

    async def __aenter__(self) -> Any:
        return await self._cm.__aenter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Any:
        return await self._cm.__aexit__(exc_type, exc_val, exc_tb)


def awaitableasynccontextmanager(
    cm: Callable[..., "AsyncContextManager[Any]"],
) -> Callable[..., _AwaitableAsyncContextManager]:
    @functools.wraps(cm)
    def f(*args: Any, **kwargs: Any) -> _AwaitableAsyncContextManager:
        return _AwaitableAsyncContextManager(cm(*args, **kwargs))

    return f
