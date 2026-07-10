import pytest
from foamlib import FoamFile
from foamlib._files._normalization import normalized
from foamlib._files._parsing import ParsedFile
from foamlib._files._serialization import dumps
from foamlib.typing import Data
from multicollections import MultiDict


def test_read() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
                entry1 value1;
                entry1 value2;
            """)
    assert parsed[("entry1",)] == "value2"


def test_read_loads() -> None:
    with pytest.warns(match="entry1"):
        assert FoamFile.loads(b"""
            entry1 value1;
            entry1 value2;
        """) == {"entry1": "value2"}


def test_read_directives() -> None:
    parsed = ParsedFile(b"""
        #directive value1
        #directive value2
    """)
    assert parsed[("#directive",)] == "value1"  # Should not overwrite or warn


def test_read_mixed() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            entry1 value1;
            entry2 value2;
            entry1 value3;
        """)
    assert parsed[("entry1",)] == "value3"
    assert parsed[("entry2",)] == "value2"


def test_read_subdictionary() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            subDict
            {
                entry1 value1;
                entry1 value2;
            }
        """)
    assert parsed[("subDict", "entry1")] == "value2"


def test_read_other() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            list (a { entry1 value1; } b { entry1 value2; entry1 value3; });
        """)
    assert parsed[("list",)] == [
        ("a", {"entry1": "value1"}),
        ("b", {"entry1": "value3"}),
    ]


def test_write_dumps() -> None:
    with pytest.warns(match="entry1"):
        FoamFile.dumps(MultiDict([("entry1", "value1"), ("entry1", "value2")]))


def test_add_directives() -> None:
    parsed = ParsedFile(b"""
        #directive value1
        #directive value2
    """)
    new_value = normalized("newValue", target=Data, keywords=("#directive",))  # ty: ignore[no-matching-overload]
    parsed.add(("#directive",), new_value, dumps(new_value))
    assert parsed[("#directive",)] == "value1"  # Should not overwrite or warn


def test_write_other() -> None:
    parsed = ParsedFile(b"""
        list (a { entry1 value1; } b { entry1 value2; });
    """)
    new_list = [
        ("a", {"entry1": "value1"}),
        ("b", MultiDict([("entry1", "value2"), ("entry1", "value3")])),
    ]
    with pytest.warns(match="entry1"):
        new_list = normalized(new_list, target=Data, keywords=("list",))  # ty: ignore[no-matching-overload]
    parsed.put(("list",), new_list, dumps(new_list))
    assert parsed[("list",)] == [
        ("a", {"entry1": "value1"}),
        ("b", {"entry1": "value3"}),
    ]
