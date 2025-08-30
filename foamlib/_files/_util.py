from __future__ import annotations

import sys
from typing import TypeVar

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

from multicollections import MultiDict

_K = TypeVar("_K")
_V = TypeVar("_V")


def as_any_dict(
    seq: Sequence[Sequence[_K | _V]] | Mapping[_K, _V],
) -> dict[_K, _V] | MultiDict[_K, _V]:
    if len(d := dict(seq)) == len(seq):  # type: ignore[arg-type]
        return d
    return MultiDict(seq)
