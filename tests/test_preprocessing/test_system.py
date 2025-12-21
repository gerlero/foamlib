"""Tests for foamlib.preprocessing.system module."""

import pytest
from foamlib.preprocessing.system import (
    control_dict,
    fv_schemes,
    fv_solution,
    simulation_parameters,
)


def test_deprecated_simulation_parameters() -> None:
    """Test deprecated simulationParameters function."""
    from foamlib.preprocessing.system import simulationParameters  # noqa: PLC0415

    with pytest.deprecated_call():
        instruction = simulationParameters(["key1", "key2"])

    assert instruction.keys == ["key1", "key2"]
    assert str(instruction.file_name) == "system/simulationParameters"


def test_deprecated_control_dict() -> None:
    """Test deprecated controlDict function."""
    from foamlib.preprocessing.system import controlDict  # noqa: PLC0415

    with pytest.deprecated_call():
        instruction = controlDict(["endTime"])

    assert instruction.keys == ["endTime"]
    assert str(instruction.file_name) == "system/controlDict"


def test_deprecated_fv_schemes() -> None:
    """Test deprecated fvSchemes function."""
    from foamlib.preprocessing.system import fvSchemes  # noqa: PLC0415

    with pytest.deprecated_call():
        instruction = fvSchemes(["ddtSchemes"])

    assert instruction.keys == ["ddtSchemes"]
    assert str(instruction.file_name) == "system/fvSchemes"


def test_deprecated_fv_solution() -> None:
    """Test deprecated fvSolution function."""
    from foamlib.preprocessing.system import fvSolution  # noqa: PLC0415

    with pytest.deprecated_call():
        instruction = fvSolution(["solvers"])

    assert instruction.keys == ["solvers"]
    assert str(instruction.file_name) == "system/fvSolution"


def test_non_deprecated_functions() -> None:
    """Test non-deprecated function versions."""
    # These should not raise deprecation warnings
    instruction1 = simulation_parameters(["key1"])
    assert instruction1.keys == ["key1"]

    instruction2 = control_dict(["endTime"])
    assert instruction2.keys == ["endTime"]

    instruction3 = fv_schemes(["ddtSchemes"])
    assert instruction3.keys == ["ddtSchemes"]

    instruction4 = fv_solution(["solvers"])
    assert instruction4.keys == ["solvers"]
