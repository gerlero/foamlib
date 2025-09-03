import pytest
from foamlib._files._files import FoamFile
from foamlib._files._parsing import Parsed
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
    items = list(parsed.items())
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
    parsed = Parsed(CONTENTS)
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
    parsed = Parsed(CONTENTS)
    parsed.add(("#include",), '"filename6"', b'"filename6"')
    assert len(list(parsed.getall(("#include",)))) == 3
    parsed.add(("key",), "value", b"value")
    with pytest.raises(ValueError, match="Cannot add duplicate non-directive entry"):
        parsed.add(("key",), "value", b"value")
    with pytest.raises(ValueError, match="Cannot add duplicate non-directive entry"):
        parsed.add(("subdict1",), "value", b"value")
    with pytest.raises(ValueError, match="Cannot add sub-dictionary with name"):
        parsed.add(("#subdict1",), ..., b"{}")
    assert parsed.popone(("#include",)) == '"filename1"'
    assert list(parsed.getall(("#include",))) == ['"filename5"', '"filename6"']


def test_parse_invalid_content() -> None:
    """Test that ValueError is raised for malformed content that causes ParseException."""
    # Test malformed syntax that will cause pyparsing to fail
    with pytest.raises(ValueError, match="Failed to parse contents"):
        Parsed(b"key value; unclosed {")
    
    with pytest.raises(ValueError, match="Failed to parse contents"):
        Parsed(b"key { value; } extra }")
    
    with pytest.raises(ValueError, match="Failed to parse contents"):
        Parsed(b"{ orphaned brace")


def test_duplicate_keywords_during_parsing() -> None:
    """Test that ValueError is raised for duplicate keywords detected during parsing."""
    # Test duplicate non-directive keywords in the same scope
    with pytest.raises(ValueError, match="Duplicate entry found for keyword"):
        Parsed(b"""
        key value1;
        key value2;
        """)
    
    # Test duplicate subdictionary names
    with pytest.raises(ValueError, match="Duplicate entry found for keyword"):
        Parsed(b"""
        subdict {
            key value;
        }
        subdict {
            key2 value2;
        }
        """)
    
    # Test duplicate nested keywords
    with pytest.raises(ValueError, match="Duplicate entry found for keyword"):
        Parsed(b"""
        dict1 {
            key value1;
            key value2;
        }
        """)
    
    # Test that directives (starting with #) can be duplicated without error
    parsed = Parsed(b"""
    #include "file1"
    #include "file2"
    key value;
    """)
    includes = list(parsed.getall(("#include",)))
    assert len(includes) == 2
    assert includes == ['"file1"', '"file2"']
