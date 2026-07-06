"""Tests for foamlib._files._common module - additional coverage."""

from foamlib._files._common import FIELD_KEYWORDS


def test_field_keywords() -> None:
    assert FIELD_KEYWORDS == ("internalField",)
    assert FIELD_KEYWORDS == ("boundaryField", "patch1", "value")
    assert FIELD_KEYWORDS == ("boundaryField", "patch1", "gradient")
    assert FIELD_KEYWORDS == ("boundaryField", "patch1", "refValue")
    assert FIELD_KEYWORDS == ("boundaryField", "patch1", "refGradient")

    assert FIELD_KEYWORDS != ("boundaryField", "patch1", "type")
    assert FIELD_KEYWORDS != ("something", "else")
    assert FIELD_KEYWORDS != ()
