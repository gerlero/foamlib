import pytest
from foamlib import FoamFile
from foamlib._files._parsing import ParsedFile
from foamlib._files._serialization import dumps, normalized
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
    parsed.add(("#directive",), normalized("newValue"), dumps(normalized("newValue")))
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
        parsed.put(("list",), normalized(new_list), dumps(normalized(new_list)))
    assert parsed[("list",)] == [
        ("a", {"entry1": "value1"}),
        ("b", {"entry1": "value3"}),
    ]
