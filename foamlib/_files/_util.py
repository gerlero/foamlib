from __future__ import annotations

import sys
from typing import Any, TypeVar

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

from multicollections import MultiDict

_K = TypeVar("_K")
_V = TypeVar("_V")


def as_any_dict(
    seq: Sequence[tuple[_K, _V]] | Mapping[_K, _V],
    *,
    recursive: bool = False,
) -> dict[_K, _V] | MultiDict[_K, _V]:
    d = dict(seq)
    result = d if len(d) == len(seq) else MultiDict(seq)

    if recursive:
        # Process values recursively for nested mappings
        if isinstance(result, dict):
            for key, value in result.items():
                result[key] = _process_value_recursively(value)
        else:  # MultiDict
            # For MultiDict, we need to process all values
            new_items = []
            for key, value in result.items():
                new_items.append((key, _process_value_recursively(value)))
            result.clear()
            result.update(new_items)

    return result


def _process_value_recursively(value: Any) -> Any:
    """Recursively process a value, converting nested mappings using as_any_dict."""
    # Check if value is a sequence of tuples that looks like a mapping
    if (isinstance(value, Sequence) and
        not isinstance(value, str) and
        (len(value) == 0 or all(isinstance(item, tuple) and len(item) == 2 for item in value))):
        # This looks like a sequence of key-value pairs (or empty), convert it
        return as_any_dict(value, recursive=True)

    # Check if value is already a mapping (dict or MultiDict)
    if isinstance(value, Mapping):
        # Convert mapping to sequence of tuples and process
        return as_any_dict(list(value.items()), recursive=True)

    # For other types (including lists that don't look like mappings), return as-is
    return value
