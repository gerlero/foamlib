"""Long lists must be parsed in linear time."""

from typing import Any

import pytest

from foamlib import loads
from foamlib._files._parsing import _parser


@pytest.mark.parametrize(
    "item",
    [
        "patch{i}",
        "6({i} 1 2 3 4 5)",
        "hex ({i} 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)",
    ],
)
def test_parsing_work_grows_linearly(
    monkeypatch: pytest.MonkeyPatch, item: str
) -> None:
    # Trying each item as "keyword value;" parsed the rest of the list every
    # time, so the number of entries parsed grew quadratically
    parsed_entries = 0
    parse_data_entry = _parser._parse_data_entry

    def counting(*args: Any) -> Any:
        nonlocal parsed_entries
        parsed_entries += 1
        return parse_data_entry(*args)

    monkeypatch.setattr(_parser, "_parse_data_entry", counting)

    def work(n: int) -> int:
        nonlocal parsed_entries
        parsed_entries = 0
        loads(f"x ({' '.join(item.format(i=i) for i in range(n))});")
        return parsed_entries

    assert work(200) < 2.5 * work(100)
