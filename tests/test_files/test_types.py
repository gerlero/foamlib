import numpy as np
import pytest
from foamlib import FoamFile


def test_dimension_set() -> None:
    dims = FoamFile.DimensionSet(length=1, time=-2)
    assert dims.length == 1
    assert dims.time == -2
    assert dims.mass == 0

    assert dims + dims == dims
    assert dims - dims == dims
    assert dims * dims == FoamFile.DimensionSet(length=2, time=-4)
    assert dims / dims == FoamFile.DimensionSet()
    assert dims**2 == FoamFile.DimensionSet(length=2, time=-4)

    assert dims
    assert not FoamFile.DimensionSet()


def test_dimensioned() -> None:
    dimensioned = FoamFile.Dimensioned(
        9.81, FoamFile.DimensionSet(length=1, time=-2), "g"
    )
    assert dimensioned.value == 9.81
    assert dimensioned.dimensions == FoamFile.DimensionSet(length=1, time=-2)
    assert dimensioned.name == "g"

    result = dimensioned + dimensioned
    assert result.value == 9.81 * 2
    assert result.dimensions == FoamFile.DimensionSet(length=1, time=-2)
    assert result.name == "g+g"

    result = dimensioned - dimensioned
    assert result.value == 0.0
    assert result.dimensions == FoamFile.DimensionSet(length=1, time=-2)
    assert result.name == "g-g"

    result = dimensioned * dimensioned
    assert result.value == 9.81**2
    assert result.dimensions == FoamFile.DimensionSet(length=2, time=-4)
    assert result.name == "g*g"

    result = dimensioned / dimensioned
    assert result.value == 1.0
    assert result.dimensions == FoamFile.DimensionSet()
    assert result.name == "g/g"

    result = dimensioned**2
    assert result.value == 9.81**2
    assert result.dimensions == FoamFile.DimensionSet(length=2, time=-4)
    assert result.name == "pow(g,2)"

    with pytest.raises(ValueError, match="dimension"):
        dimensioned + 1

    with pytest.raises(ValueError, match="dimension"):
        float(dimensioned)

    with pytest.raises(ValueError, match="dimension"):
        np.array(dimensioned)
