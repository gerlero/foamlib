"""Tests for foamlib.preprocessing.parameter_study module - additional coverage."""

import pytest
from foamlib.preprocessing.parameter_study import record_generator


def test_record_generator_empty_records() -> None:
    """Test record_generator with empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot generate ParameterStudy from empty list"):
        record_generator(
            records=[],
            template_case="tests/test_preprocessing/templates/damBreak",
            output_folder="Cases"
        )
