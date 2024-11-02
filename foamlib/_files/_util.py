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
    from ._base import FoamFileBase


def is_sequence(
    value: FoamFileBase.Data,
) -> TypeGuard[Sequence[FoamFileBase.Data]]:
    return isinstance(value, Sequence) and not isinstance(value, str)
