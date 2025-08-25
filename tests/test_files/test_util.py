"""Tests for foamlib._files._util multimapping classes.

These tests verify that MultiMapping and MutableMultiMapping classes
behave like multidicts from the multidict library.
"""

from __future__ import annotations

from typing import Iterator, TypeVar

import multidict
import pytest
from foamlib._files._util import MultiMapping, MutableMultiMapping

K = TypeVar("K")
V = TypeVar("V")


class MultiFrozenDict(MultiMapping[str, str]):
    """Concrete implementation of MultiMapping for testing."""

    def __init__(self, data: list[tuple[str, str]] | None = None) -> None:
        if data is None:
            data = []
        self._data: list[tuple[str, str]] = data

    def _getall(self, key: str) -> list[str]:
        return [value for k, value in self._data if k == key]

    def __iter__(self) -> Iterator[str]:
        # Return all keys including duplicates, like multidict does
        for key, _ in self._data:
            yield key

    def __len__(self) -> int:
        return len(self._data)


class MultiDict(MutableMultiMapping[str, str]):
    """Concrete implementation of MutableMultiMapping for testing."""

    def __init__(self, data: list[tuple[str, str]] | None = None) -> None:
        if data is None:
            data = []
        self._data: list[tuple[str, str]] = data

    def _getall(self, key: str) -> list[str]:
        return [value for k, value in self._data if k == key]

    def __iter__(self) -> Iterator[str]:
        # Return all keys including duplicates, like multidict does
        for key, _ in self._data:
            yield key

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: str) -> None:
        # Remove all existing values for this key, then add the new one
        self._data = [(k, v) for k, v in self._data if k != key]
        self._data.append((key, value))

    def add(self, key: str, value: str) -> None:
        self._data.append((key, value))

    def _popone(self, key: str) -> str:
        for i, (k, v) in enumerate(self._data):
            if k == key:
                self._data.pop(i)
                return v
        raise KeyError(key)


def test_multimapping_basic_functionality() -> None:
    """Test basic MultiMapping functionality against multidict behavior."""
    # Test data with duplicate keys
    data = [("key1", "value1"), ("key1", "value2"), ("key2", "value3")]

    # Create our test implementation
    test_mm = MultiFrozenDict(data)

    # Create multidict for comparison
    md = multidict.MultiDict(data)

    # Test getone - should return first value
    assert test_mm.getone("key1") == md.getone("key1")
    assert test_mm.getone("key2") == md.getone("key2")

    # Test getall - should return all values for key
    assert test_mm.getall("key1") == md.getall("key1")
    assert test_mm.getall("key2") == md.getall("key2")

    # Test __getitem__ - should behave like getone
    assert test_mm["key1"] == md["key1"]
    assert test_mm["key2"] == md["key2"]

    # Test length
    assert len(test_mm) == len(md)

    # Test iteration over keys
    assert list(test_mm) == list(md.keys())


def test_multimapping_missing_keys() -> None:
    """Test behavior with missing keys."""
    test_mm = MultiFrozenDict([("key1", "value1")])
    md = multidict.MultiDict([("key1", "value1")])

    # Test KeyError for missing keys
    with pytest.raises(KeyError):
        test_mm.getone("missing")

    with pytest.raises(KeyError):
        md.getone("missing")

    # Test getone with default
    assert test_mm.getone("missing", "default") == "default"
    assert md.getone("missing", "default") == "default"

    # Test getall with missing key
    with pytest.raises(KeyError):
        test_mm.getall("missing")

    # Test getall with default
    assert test_mm.getall("missing", "default") == "default"
    assert md.getall("missing", "default") == "default"


def test_mutable_multimapping_add() -> None:
    """Test MutableMultiMapping.add() method."""
    test_mmm = MultiDict()
    md: multidict.MultiDict[str] = multidict.MultiDict()

    # Add some values
    test_mmm.add("key1", "value1")
    test_mmm.add("key1", "value2")
    test_mmm.add("key2", "value3")

    md.add("key1", "value1")
    md.add("key1", "value2")
    md.add("key2", "value3")

    # Test that both have same behavior
    assert test_mmm.getall("key1") == md.getall("key1")
    assert test_mmm.getall("key2") == md.getall("key2")
    assert len(test_mmm) == len(md)


def test_mutable_multimapping_popall() -> None:
    """Test MutableMultiMapping.popall() method."""
    data = [("key1", "value1"), ("key1", "value2"), ("key2", "value3")]
    test_mmm = MultiDict(data)

    # Test popall returns all values and removes them
    values = test_mmm.popall("key1")
    assert values == ["value1", "value2"]
    assert test_mmm.getall("key1", []) == []
    assert test_mmm.getall("key2") == ["value3"]

    # Test popall on missing key
    with pytest.raises(KeyError):
        test_mmm.popall("missing")

    # Test popall with default
    result = test_mmm.popall("missing", "default")
    assert result == "default"


def test_mutable_multimapping_extend() -> None:
    """Test MutableMultiMapping.extend() method."""
    test_mmm = MultiDict([("key1", "value1")])

    # Test extend with mapping
    test_mmm.extend({"key2": "value2", "key3": "value3"})
    assert test_mmm.getone("key2") == "value2"
    assert test_mmm.getone("key3") == "value3"

    # Test extend with iterable of tuples
    test_mmm.extend([("key1", "value2"), ("key4", "value4")])
    assert test_mmm.getall("key1") == ["value1", "value2"]
    assert test_mmm.getone("key4") == "value4"

    # Test extend with kwargs
    test_mmm.extend(key5="value5", key6="value6")
    assert test_mmm.getone("key5") == "value5"
    assert test_mmm.getone("key6") == "value6"


def test_mutable_multimapping_merge() -> None:
    """Test MutableMultiMapping.merge() method."""
    test_mmm = MultiDict([("key1", "value1")])

    # Test merge with mapping (should not overwrite existing keys)
    test_mmm.merge({"key1": "value2", "key2": "value3"})
    assert test_mmm.getall("key1") == ["value1"]  # Should not change
    assert test_mmm.getone("key2") == "value3"  # Should be added

    # Test merge with iterable of tuples
    test_mmm.merge([("key1", "value4"), ("key3", "value5")])
    assert test_mmm.getall("key1") == ["value1"]  # Should not change
    assert test_mmm.getone("key3") == "value5"  # Should be added

    # Test merge with kwargs
    test_mmm.merge(key1="value6", key4="value7")
    assert test_mmm.getall("key1") == ["value1"]  # Should not change
    assert test_mmm.getone("key4") == "value7"  # Should be added


def test_mutable_multimapping_standard_mapping_interface() -> None:
    """Test that MutableMultiMapping supports standard mapping interface."""
    test_mmm = MultiDict()

    # Test setitem
    test_mmm["key1"] = "value1"
    assert test_mmm["key1"] == "value1"

    # Test setitem overwrites existing values
    test_mmm.add("key1", "value2")
    assert test_mmm.getall("key1") == ["value1", "value2"]
    test_mmm["key1"] = "value3"
    assert test_mmm.getall("key1") == ["value3"]

    # Test delitem
    test_mmm.add("key2", "value4")
    assert "key2" in test_mmm
    del test_mmm["key2"]
    assert "key2" not in test_mmm

    # Test delitem on missing key
    with pytest.raises(KeyError):
        del test_mmm["missing"]


def test_multimapping_empty() -> None:
    """Test behavior with empty multimapping."""
    test_mm = MultiFrozenDict()
    test_mmm = MultiDict()

    assert len(test_mm) == 0
    assert len(test_mmm) == 0
    assert list(test_mm) == []
    assert list(test_mmm) == []

    # Test that missing keys behave correctly
    with pytest.raises(KeyError):
        test_mm["missing"]


def test_multimapping_none_type_handling() -> None:
    """Test that _NoneType is handled correctly in getone and getall."""
    test_mm = MultiFrozenDict([("key1", "value1")])

    # Test that None can be used as a default value (should be distinct from _NONE)
    result = test_mm.getone("missing", None)
    assert result is None

    result2 = test_mm.getall("missing", None)
    assert result2 is None

    # Test with actual values
    result3 = test_mm.getone("missing", "default")
    assert result3 == "default"

    result4 = test_mm.getall("missing", [])
    assert result4 == []


def test_mutable_multimapping_popall_edge_cases() -> None:
    """Test edge cases for popall method."""
    test_mmm = MultiDict([("key1", "value1")])

    # Test popall with single value
    values = test_mmm.popall("key1")
    assert values == ["value1"]
    assert len(test_mmm) == 0

    # Test popall on empty mapping
    with pytest.raises(KeyError):
        test_mmm.popall("missing")

    # Test popall with None as default (should work)
    result = test_mmm.popall("missing", None)
    assert result is None


def test_comprehensive_multidict_comparison() -> None:
    """Comprehensive test comparing behavior with multidict library."""
    # Test various operations side by side
    data = [("a", "1"), ("b", "2"), ("a", "3"), ("c", "4"), ("b", "5")]

    test_mm = MultiFrozenDict(data)
    test_mmm = MultiDict(data.copy())
    md = multidict.MultiDict(data)

    # Test basic operations
    assert test_mm.getone("a") == md.getone("a")
    assert test_mm.getall("a") == md.getall("a")
    assert test_mm.getall("b") == md.getall("b")
    assert test_mm.getone("c") == md.getone("c")

    # Test iteration
    assert list(test_mm) == list(md)
    assert list(test_mmm) == list(md)

    # Test length
    assert len(test_mm) == len(md)
    assert len(test_mmm) == len(md)

    # Test mutable operations
    test_mmm.add("d", "6")
    md.add("d", "6")
    assert test_mmm.getall("d") == md.getall("d")
    assert len(test_mmm) == len(md)

    # Test popall equivalent
    test_values = test_mmm.popall("a")
    md_values = md.popall("a")
    assert test_values == md_values
    assert len(test_mmm) == len(md)
