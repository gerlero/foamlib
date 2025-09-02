"""Tests for the _util.as_any_dict function."""
from __future__ import annotations

from foamlib._files._util import as_any_dict
from multicollections import MultiDict


def test_as_any_dict_simple_dict() -> None:
    """Test basic dict functionality without duplicates."""
    seq = [("a", 1), ("b", 2), ("c", 3)]
    result = as_any_dict(seq)
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_as_any_dict_multidict() -> None:
    """Test MultiDict functionality with duplicates."""
    seq = [("a", 1), ("b", 2), ("a", 3)]
    result = as_any_dict(seq)
    assert isinstance(result, MultiDict)
    assert list(result.items()) == [("a", 1), ("b", 2), ("a", 3)]


def test_as_any_dict_non_recursive_nested() -> None:
    """Test that nested structures are not processed without recursive=True."""
    seq = [
        ("outer1", [("inner1", 1), ("inner2", 2)]),
        ("outer2", [("inner3", 3), ("inner1", 4)])
    ]
    result = as_any_dict(seq)
    assert isinstance(result, dict)
    assert result["outer1"] == [("inner1", 1), ("inner2", 2)]
    assert result["outer2"] == [("inner3", 3), ("inner1", 4)]


def test_as_any_dict_recursive_nested() -> None:
    """Test recursive processing of nested structures."""
    seq = [
        ("outer1", [("inner1", 1), ("inner2", 2)]),
        ("outer2", [("inner3", 3), ("inner1", 4)])
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["outer1"], dict)
    assert isinstance(result["outer2"], dict)
    assert result["outer1"] == {"inner1": 1, "inner2": 2}
    assert result["outer2"] == {"inner3": 3, "inner1": 4}


def test_as_any_dict_recursive_inner_duplicates() -> None:
    """Test recursive processing with duplicates in nested structures."""
    seq = [
        ("outer1", [("inner", 1), ("inner", 2)]),  # inner duplicated
        ("outer2", [("inner3", 3), ("inner4", 4)])
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["outer1"], MultiDict)
    assert isinstance(result["outer2"], dict)
    assert list(result["outer1"].items()) == [("inner", 1), ("inner", 2)]
    assert result["outer2"] == {"inner3": 3, "inner4": 4}


def test_as_any_dict_recursive_outer_duplicates() -> None:
    """Test recursive processing with duplicates at outer level."""
    seq = [
        ("outer", [("inner1", 1), ("inner2", 2)]),
        ("outer", [("inner3", 3), ("inner4", 4)])  # outer duplicated
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, MultiDict)
    values = result.getall("outer")
    assert len(values) == 2
    assert all(isinstance(v, dict) for v in values)
    assert values[0] == {"inner1": 1, "inner2": 2}
    assert values[1] == {"inner3": 3, "inner4": 4}


def test_as_any_dict_recursive_deeply_nested() -> None:
    """Test deeply nested recursive processing."""
    seq = [
        ("level1", [
            ("level2a", [("level3", 1), ("level3b", 2)]),
            ("level2b", [("level3c", 3), ("level3c", 4)])  # level3c duplicated
        ])
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["level1"], dict)
    assert isinstance(result["level1"]["level2a"], dict)
    assert isinstance(result["level1"]["level2b"], MultiDict)

    assert result["level1"]["level2a"] == {"level3": 1, "level3b": 2}
    assert list(result["level1"]["level2b"].items()) == [("level3c", 3), ("level3c", 4)]


def test_as_any_dict_recursive_mixed_types() -> None:
    """Test recursive processing with mixed value types."""
    seq = [
        ("string_val", "hello"),
        ("number_val", 42),
        ("dict_val", [("a", 1), ("b", 2)]),
        ("list_val", [1, 2, 3])  # regular list, not key-value pairs
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert result["string_val"] == "hello"
    assert result["number_val"] == 42
    assert isinstance(result["dict_val"], dict)
    assert result["dict_val"] == {"a": 1, "b": 2}
    assert result["list_val"] == [1, 2, 3]  # unchanged


def test_as_any_dict_recursive_existing_mapping() -> None:
    """Test recursive processing of existing mappings."""
    # Test with existing dict
    mapping = {"outer": {"inner1": 1, "inner2": 2}}
    result = as_any_dict(mapping.items(), recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["outer"], dict)
    assert result["outer"] == {"inner1": 1, "inner2": 2}


def test_as_any_dict_empty() -> None:
    """Test with empty sequence."""
    result = as_any_dict([])
    assert isinstance(result, dict)
    assert result == {}

    result_recursive = as_any_dict([], recursive=True)
    assert isinstance(result_recursive, dict)
    assert result_recursive == {}


def test_as_any_dict_edge_cases() -> None:
    """Test edge cases for recursive processing."""
    # Empty nested sequences
    seq = [("outer", [])]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["outer"], dict)
    assert result["outer"] == {}

    # Single item nested
    seq = [("outer", [("inner", 1)])]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["outer"], dict)
    assert result["outer"] == {"inner": 1}

    # Nested sequences that are not key-value pairs
    seq = [
        ("valid_dict", [("a", 1), ("b", 2)]),
        ("invalid_list", [(1, 2, 3)]),  # tuples with wrong length
        ("invalid_list2", [1, 2, 3])  # not tuples at all
    ]
    result = as_any_dict(seq, recursive=True)
    assert isinstance(result, dict)
    assert isinstance(result["valid_dict"], dict)
    assert result["valid_dict"] == {"a": 1, "b": 2}
    assert result["invalid_list"] == [(1, 2, 3)]  # unchanged
    assert result["invalid_list2"] == [1, 2, 3]  # unchanged
