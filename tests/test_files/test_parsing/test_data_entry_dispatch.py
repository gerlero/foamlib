"""A data entry is only tried with the parsers that can accept its first byte."""

from typing import Any

import pytest

from foamlib import loads
from foamlib._files._parsing import _parser


def test_parsers_that_cannot_match_are_not_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def counting(name: str) -> Any:
        parser = getattr(_parser, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return parser(*args, **kwargs)

        return wrapper

    for name in (
        "_parse_field",
        "_parse_typed_list",
        "_parse_dimensioned",
        "_parse_dimensions",
    ):
        monkeypatch.setattr(_parser, name, counting(name))

    items = " ".join(["1 2.5 word yes (3 4) -1e-3"] * 50)
    assert loads(f"x ({items});") == {"x": [1, 2.5, "word", True, [3, 4], -1e-3] * 50}
    assert calls == []


def test_each_kind_of_entry_is_still_read() -> None:
    parsed: Any = loads(
        "a 3{1.5}; b 2(1 2); c nu [0 2 -1 0 0 0 0] 1e-5; d [0 1 0 0 0 0 0];"
        " e uniform (1 2 3); f List<label> 2(1 2); g inf; h off; i 2 {k 1;};"
    )
    assert parsed["a"] == [1.5, 1.5, 1.5]
    assert parsed["b"] == [1, 2]
    assert (parsed["c"].name, parsed["c"].value) == ("nu", 1e-5)
    assert parsed["d"].length == 1
    assert parsed["e"].tolist() == [1, 2, 3]
    assert parsed["f"].tolist() == [1, 2]
    assert parsed["g"] == float("inf")
    assert parsed["h"] is False
    assert parsed["i"] == [("k", 1), ("k", 1)]
