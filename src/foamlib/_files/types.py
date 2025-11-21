from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from ._util import is_sequence

if TYPE_CHECKING:
    from ._typing import Tensor, TensorLike


class DimensionSet(NamedTuple):
    """Set of physical dimensions represented as powers of base SI units.

    Corresponds to the `dimensionSet` type in OpenFOAM.

    :param mass: Power of the mass dimension.
    :param length: Power of the length dimension.
    :param time: Power of the time dimension.
    :param temperature: Power of the temperature dimension.
    :param moles: Power of the amount of substance dimension.
    :param current: Power of the electric current dimension.
    :param luminous_intensity: Power of the luminous intensity dimension.
    """

    mass: float = 0
    length: float = 0
    time: float = 0
    temperature: float = 0
    moles: float = 0
    current: float = 0
    luminous_intensity: float = 0

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self) if v != 0)})"

    @override
    def __add__(self, other: DimensionSet, /) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot add DimensionSet with different dimensions: {self} + {other}"
            raise ValueError(msg)

        return self

    def __sub__(self, other: DimensionSet, /) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot subtract DimensionSet with different dimensions: {self} - {other}"
            raise ValueError(msg)

        return self

    @override
    def __mul__(self, other: DimensionSet, /) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a + b for a, b in zip(self, other)))

    def __truediv__(self, other: DimensionSet, /) -> DimensionSet:
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a - b for a, b in zip(self, other)))

    def __pow__(self, exponent: float, /) -> DimensionSet:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return DimensionSet(*(a * exponent for a in self))

    def __bool__(self) -> bool:
        return any(v != 0 for v in self)


class Dimensioned:
    """A numerical value with associated physical dimensions.

    Corresponds to the `dimensioned<...>` type in OpenFOAM.

    The `value` can be a single number (scalar) or a 1D array (vector or tensor).

    :param value: The numerical value.
    :param dimensions: The physical dimensions as a :class:`DimensionSet` or a sequence of up to 7 numbers.
    :param name: An optional name for the dimensioned quantity.
    """

    def __init__(
        self,
        value: TensorLike,
        dimensions: DimensionSet | Sequence[float],
        name: str | None = None,
    ) -> None:
        if is_sequence(value):
            self.value: Tensor = np.array(value, dtype=float)
        else:
            assert isinstance(value, (int, float, np.ndarray))
            self.value = float(value)

        if not isinstance(dimensions, DimensionSet):
            self.dimensions = DimensionSet(*dimensions)
        else:
            self.dimensions = dimensions

        self.name = name

    @override
    def __repr__(self) -> str:
        if self.name is not None:
            return (
                f"{type(self).__name__}({self.value}, {self.dimensions}, {self.name})"
            )
        return f"{type(self).__name__}({self.value}, {self.dimensions})"

    def __add__(self, other: Dimensioned | Tensor, /) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value + other.value,
            self.dimensions + other.dimensions,
            f"{self.name}+{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __sub__(self, other: Dimensioned | Tensor, /) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value - other.value,
            self.dimensions - other.dimensions,
            f"{self.name}-{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __mul__(self, other: Dimensioned | Tensor, /) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value * other.value,
            self.dimensions * other.dimensions,
            f"{self.name}*{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __truediv__(self, other: Dimensioned | Tensor, /) -> Dimensioned:
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value / other.value,
            self.dimensions / other.dimensions,
            f"{self.name}/{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __pow__(self, exponent: float, /) -> Dimensioned:
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return Dimensioned(
            self.value**exponent,
            self.dimensions**exponent,
            f"pow({self.name},{exponent})" if self.name is not None else None,
        )

    def __float__(self) -> float:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to float: {self.dimensions}"
            raise ValueError(msg)
        return float(self.value)

    def __int__(self) -> int:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to int: {self.dimensions}"
            raise ValueError(msg)
        return int(self.value)

    def __array__(
        self, dtype: Any = None, *, copy: Any = None
    ) -> np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64]]:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to array: {self.dimensions}"
            raise ValueError(msg)
        return np.array(self.value, dtype=dtype, copy=copy)
