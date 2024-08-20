import sys
from typing import Any

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


def is_sequence(
    value: Any,
) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, str)
