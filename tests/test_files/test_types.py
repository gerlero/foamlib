import numpy as np
import pytest
from foamlib import Dimensioned, DimensionSet


def test_dimension_set() -> None:
    dims = DimensionSet(length=1, time=-2)
    assert dims.length == 1
    assert dims.time == -2
    assert dims.mass == 0

    assert dims + dims == dims
    assert dims - dims == dims
    assert dims * dims == DimensionSet(length=2, time=-4)
    assert dims / dims == DimensionSet()
    assert dims**2 == DimensionSet(length=2, time=-4)

    assert dims
    assert not DimensionSet()


def test_dimension_set_not_implemented() -> None:
    """Test DimensionSet operations with incompatible types return NotImplemented."""
    dims = DimensionSet(length=1, time=-2)

    # Test __add__ with non-DimensionSet
    assert dims.__add__("string") == NotImplemented

    # Test __sub__ with non-DimensionSet
    assert dims.__sub__("string") == NotImplemented

    # Test __mul__ with non-DimensionSet
    assert dims.__mul__("string") == NotImplemented

    # Test __truediv__ with non-DimensionSet
    assert dims.__truediv__("string") == NotImplemented

    # Test __pow__ with non-number
    assert dims.__pow__("string") == NotImplemented


def test_dimensioned() -> None:
    dimensioned = Dimensioned(9.81, DimensionSet(length=1, time=-2), "g")
    assert dimensioned.value == 9.81
    assert dimensioned.dimensions == DimensionSet(length=1, time=-2)
    assert dimensioned.name == "g"

    result = dimensioned + dimensioned
    assert result.value == 9.81 * 2
    assert result.dimensions == DimensionSet(length=1, time=-2)
    assert result.name == "g+g"

    result = dimensioned - dimensioned
    assert result.value == 0.0
    assert result.dimensions == DimensionSet(length=1, time=-2)
    assert result.name == "g-g"

    result = dimensioned * dimensioned
    assert result.value == 9.81**2
    assert result.dimensions == DimensionSet(length=2, time=-4)
    assert result.name == "g*g"

    result = dimensioned / dimensioned
    assert result.value == 1.0
    assert result.dimensions == DimensionSet()
    assert result.name == "g/g"

    result = dimensioned**2
    assert result.value == 9.81**2
    assert result.dimensions == DimensionSet(length=2, time=-4)
    assert result.name == "pow(g,2)"

    with pytest.raises(ValueError, match="dimension"):
        dimensioned + 1

    with pytest.raises(ValueError, match="dimension"):
        float(dimensioned)

    with pytest.raises(ValueError, match="dimension"):
        np.array(dimensioned)


def test_dimensioned_invalid_array() -> None:
    """Test Dimensioned with invalid numpy array shapes."""
    # Array with wrong shape
    with pytest.raises(ValueError, match="Invalid array"):
        Dimensioned(np.array([1, 2]), DimensionSet())

    # Array with wrong dtype but valid shape should work
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = Dimensioned(arr, DimensionSet())
    assert isinstance(d.value, np.ndarray)


def test_dimensioned_invalid_sequence() -> None:
    """Test Dimensioned with invalid sequences."""
    # Wrong number of elements
    with pytest.raises(ValueError, match="Invalid sequence"):
        Dimensioned([1, 2], DimensionSet())

    with pytest.raises(ValueError, match="Invalid sequence"):
        Dimensioned([1, 2, 3, 4], DimensionSet())


def test_dimensioned_invalid_type() -> None:
    """Test Dimensioned with invalid types."""
    with pytest.raises(TypeError, match="Invalid type"):
        Dimensioned("string", DimensionSet())


def test_dimensioned_invalid_name_type() -> None:
    """Test Dimensioned with invalid name type."""
    with pytest.raises(TypeError, match="Invalid type for Dimensioned name"):
        Dimensioned(1.0, DimensionSet(), name=123)  # type: ignore[arg-type]


def test_dimensioned_invalid_name_value() -> None:
    """Test Dimensioned with invalid name value (not parseable due to spaces)."""
    from foamlib._files._parsing.exceptions import FoamFileDecodeError  # noqa: PLC0415

    # Name with spaces cannot be parsed as a valid OpenFOAM token
    with pytest.raises(FoamFileDecodeError):
        Dimensioned(1.0, DimensionSet(), name="invalid name with spaces")


def test_dimensioned_with_sequence_init() -> None:
    """Test Dimensioned with dimension sequence instead of DimensionSet."""
    # Create with tuple of dimensions
    d = Dimensioned(1.0, (1, 0, -2, 0, 0, 0, 0))
    assert d.dimensions == DimensionSet(mass=1, time=-2)
