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
