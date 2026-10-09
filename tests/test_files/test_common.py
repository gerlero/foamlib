"""Tests for foamlib._files._common module - additional coverage."""

from foamlib._files._common import expect_field


def test_expect_field() -> None:
    assert expect_field(("internalField",))
    assert expect_field(("boundaryField", "patch1", "value"))
    assert expect_field(("boundaryField", "patch1", "gradient"))
    assert expect_field(("boundaryField", "patch1", "refValue"))
    assert expect_field(("boundaryField", "patch1", "refGradient"))

    assert not expect_field(("boundaryField", "patch1", "type"))
    assert not expect_field(("something", "else"))
    assert not expect_field(())
