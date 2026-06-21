"""Tests for foamlib._files._util module - additional coverage."""

from foamlib._files._util import add_to_mapping
from multicollections import MultiDict


def test_add_to_mapping_multidict_add() -> None:
    """Test add_to_mapping with a MutableMultiMapping adds the value."""
    d = MultiDict[str, int]()
    d["key1"] = 1

    result = add_to_mapping(d, "key1", 2)

    assert isinstance(result, MultiDict)
    assert list(result.getall("key1")) == [1, 2]


def test_add_to_mapping_dict_converts_to_multidict() -> None:
    """Test add_to_mapping converts dict to MultiDict when adding duplicate key."""
    d: dict[str, int] = {"key1": 1}

    result = add_to_mapping(d, "key1", 2)

    assert isinstance(result, MultiDict)
    assert list(result.getall("key1")) == [1, 2]


def test_add_to_mapping_dict_new_key() -> None:
    """Test add_to_mapping with dict and new key returns the same dict."""
    d: dict[str, int] = {"key1": 1}

    result = add_to_mapping(d, "key2", 2)

    assert isinstance(result, dict)
    assert result is d
    assert result["key1"] == 1
    assert result["key2"] == 2
