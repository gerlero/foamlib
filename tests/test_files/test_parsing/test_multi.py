import pytest
from foamlib import FoamFile
from foamlib._files._parsing import ParsedFile
from foamlib._files._serialization import normalized
from multicollections import MultiDict

CONTENTS = b"""
#include "filename1"
subdict1 {
#include "filename2"
key value;
#include "filename3"
}
subdict2 {
#include "filename4"
key2 value2;
}
#include "filename5"
"""


def test_loads() -> None:
    parsed = FoamFile.loads(CONTENTS)
    assert isinstance(parsed, MultiDict)
    items = list(parsed.items())  # ty: ignore[invalid-argument-type]
    assert len(items) == 4
    assert items[0] == ("#include", '"filename1"')
    assert items[1][0] == "subdict1"
    assert isinstance(items[1][1], MultiDict)
    subdict1_items = list(items[1][1].items())
    assert len(subdict1_items) == 3
    assert subdict1_items[0] == ("#include", '"filename2"')
    assert subdict1_items[1] == ("key", "value")
    assert subdict1_items[2] == ("#include", '"filename3"')
    assert items[2][0] == "subdict2"
    assert isinstance(items[2][1], dict)
    subdict2_items = list(items[2][1].items())
    assert len(subdict2_items) == 2
    assert subdict2_items[0] == ("#include", '"filename4"')
    assert subdict2_items[1] == ("key2", "value2")
    assert items[3] == ("#include", '"filename5"')


def test_parsed_as_dict() -> None:
    parsed = ParsedFile(CONTENTS)
    d = parsed.as_dict()
    assert isinstance(d, MultiDict)
    items = list(d.items())
    assert len(items) == 4
    assert items[0] == ("#include", '"filename1"')
    assert items[1][0] == "subdict1"
    assert isinstance(items[1][1], MultiDict)
    subdict1_items = list(items[1][1].items())
    assert len(subdict1_items) == 3
    assert subdict1_items[0] == ("#include", '"filename2"')
    assert subdict1_items[1] == ("key", "value")
    assert subdict1_items[2] == ("#include", '"filename3"')
    assert items[2][0] == "subdict2"
    assert isinstance(items[2][1], dict)
    subdict2_items = list(items[2][1].items())
    assert len(subdict2_items) == 2
    assert subdict2_items[0] == ("#include", '"filename4"')
    assert subdict2_items[1] == ("key2", "value2")
    assert items[3] == ("#include", '"filename5"')


def test_parsed_mutation() -> None:
    parsed = ParsedFile(CONTENTS)
    parsed.add(("#include",), '"filename6"', b'"filename6"')
    assert len(list(parsed.getall(("#include",)))) == 3
    parsed.add(("key",), "value", b"value")
    with pytest.raises(AssertionError):
        parsed.add(("key",), "value", b"value")
    with pytest.raises(AssertionError):
        parsed.add(("subdict1",), "value", b"value")
    with pytest.raises(AssertionError):
        parsed.add(("#subdict1",), ..., b"{}")
    assert parsed.popone(("#include",)) == '"filename1"'
    assert list(parsed.getall(("#include",))) == ['"filename5"', '"filename6"']


def test_invalid_duplicate_keywords() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ParsedFile(b"""
        key value1;
        key value2;
        """)

    with pytest.raises(ValueError, match="Duplicate"):
        normalized(MultiDict([("key", "value1"), ("key", "value2")]), keywords=())

    with pytest.raises(ValueError, match="Duplicate"):
        ParsedFile(b"""
        subdict {
            key value;
        }
        subdict {
            key2 value2;
        }
        """)

    with pytest.raises(ValueError, match="Duplicate"):
        normalized(
            MultiDict(
                [
                    ("subdict", {"key": "value"}),
                    ("subdict", {"key2": "value2"}),
                ]
            ),
            keywords=(),
        )

    with pytest.raises(ValueError, match="Duplicate"):
        ParsedFile(b"""
        dict1 {
            key value1;
            key value2;
        }
        """)

    with pytest.raises(ValueError, match="Duplicate"):
        normalized(
            {"dict1": MultiDict([("key", "value1"), ("key", "value2")])},
            keywords=(),
        )

    with pytest.raises(ValueError, match="Duplicate"):
        ParsedFile(b"""
        list (subdict { a b; a c; });
        """)

    with pytest.raises(ValueError, match="Duplicate"):
        normalized(
            {"list": [("subdict", MultiDict([("a", "b"), ("a", "c")]))]}, keywords=()
        )
