from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

if TYPE_CHECKING:
    from ._types import Data


def is_sequence(
    value: Data,
) -> TypeGuard[Sequence[Data]]:
    return isinstance(value, Sequence) and not isinstance(value, str)
