"""Tests for foamlib._files._common module - additional coverage."""

from foamlib._files._common import _expect_field


def test_expect_field_edge_cases() -> None:
    """Test _expect_field with various edge cases."""
    # Test with internalField
    assert _expect_field(("internalField",)) is True

    # Test with boundaryField value
    assert _expect_field(("boundaryField", "patch1", "value")) is True

    # Test with boundaryField gradient
    assert _expect_field(("boundaryField", "patch1", "gradient")) is True

    # Test with keywords ending in Value
    assert _expect_field(("boundaryField", "patch1", "refValue")) is True

    # Test with keywords ending in Gradient
    assert _expect_field(("boundaryField", "patch1", "refGradient")) is True

    # Test with non-matching keyword
    assert _expect_field(("boundaryField", "patch1", "type")) is False

    # Test with different tuple structure
    assert _expect_field(("something", "else")) is False

    # Test with non-tuple
    assert _expect_field("internalField") is False

    # Test with empty tuple
    assert _expect_field(()) is False
