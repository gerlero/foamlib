import io

import pytest
from multicollections import MultiDict

from foamlib import FoamFile, loads
from foamlib._files._encoding import encoded
from foamlib._files._parsing import ParsedFile
from foamlib.typing import Data, StandaloneData


def test_read() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
                entry1 value1;
                entry1 value2;
            """)
    assert len(parsed) == 1
    assert parsed[("entry1",)] == "value2"


def test_read_loads() -> None:
    with pytest.warns(match="entry1"):
        assert loads(b"""
            entry1 value1;
            entry1 value2;
        """) == {"entry1": "value2"}


def test_read_directives() -> None:
    parsed = ParsedFile(b"""
        #directive value1
        #directive value2
    """)
    assert len(parsed) == 2
    assert parsed[("#directive",)] == "value1"
    assert parsed.getall(("#directive",)) == ["value1", "value2"]


def test_read_mixed() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            entry1 value1;
            entry2 value2;
            entry1 value3;
        """)
    assert len(parsed) == 2
    assert list(parsed.as_dict().items()) == [
        ("entry2", "value2"),
        ("entry1", "value3"),
    ]


def test_read_subdictionary() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            subDict
            {
                entry1 value1;
                entry1 value2;
            }
        """)
    assert len(parsed) == 2
    assert parsed[("subDict", "entry1")] == "value2"


def test_read_other() -> None:
    with pytest.warns(match="entry1"):
        parsed = ParsedFile(b"""
            list (a { entry1 value1; } b { entry1 value2; entry1 value3; });
        """)
    assert len(parsed) == 1
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
    assert len(parsed) == 2
    assert parsed[("#directive",)] == "value1"
    assert parsed.getall(("#directive",)) == ["value1", "value2"]
    new_value = encoded("newValue", target=Data, keywords=("#directive",))
    with io.BytesIO() as f:
        encoded(new_value, f, target=StandaloneData)
        parsed.add(("#directive",), new_value, f.getvalue())
    assert parsed[("#directive",)] == "value1"  # Should not overwrite or warn


def test_write_other() -> None:
    parsed = ParsedFile(b"""
        list (a { entry1 value1; } b { entry1 value2; });
    """)
    assert len(parsed) == 1
    new_list = [
        ("a", {"entry1": "value1"}),
        ("b", MultiDict([("entry1", "value2"), ("entry1", "value3")])),
    ]
    with pytest.warns(match="entry1"):
        new_list = encoded(new_list, target=Data, keywords=("list",))
    with io.BytesIO() as f:
        encoded(new_list, f, target=StandaloneData)
        parsed.put(("list",), new_list, f.getvalue())
    assert parsed[("list",)] == [
        ("a", {"entry1": "value1"}),
        ("b", {"entry1": "value3"}),
    ]
