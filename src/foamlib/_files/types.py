import sys
from typing import TYPE_CHECKING, Literal, Self, TypeVar, overload

import numpy as np

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from ..typing import DimensionSetLike, Tensor, TensorLike


_T = TypeVar("_T", bound=np.floating | np.integer)


class DimensionSet(
    tuple[
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
    ]
):
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

    __slots__ = ()

    _fields = (
        "mass",
        "length",
        "time",
        "temperature",
        "moles",
        "current",
        "luminous_intensity",
    )

    @overload
    def __new__(
        cls,
        mass: int | float,
        length: int | float,
        time: int | float,
        temperature: int | float,
        moles: int | float,
        current: int | float = 0,
        luminous_intensity: int | float = 0,
        /,
    ) -> Self: ...

    @overload
    def __new__(
        cls,
        *,
        mass: int | float = 0,
        length: int | float = 0,
        time: int | float = 0,
        temperature: int | float = 0,
        moles: int | float = 0,
        current: int | float = 0,
        luminous_intensity: int | float = 0,
    ) -> Self: ...

    def __new__(
        cls,
        *args: int | float,
        **kwargs: int | float,
    ) -> Self:
        if args and kwargs:
            msg = "Cannot mix positional and keyword arguments for DimensionSet"
            raise TypeError(msg)

        if args:
            match args:
                case [*_] if 5 <= len(args) <= 7:
                    values = list(args)
                case _:
                    msg = f"DimensionSet positional constructor requires 5 to 7 arguments, got {len(args)}"
                    raise TypeError(msg)
        else:
            for name in kwargs:
                if name not in cls._fields:
                    msg = f"Invalid keyword argument for DimensionSet: {name}"
                    raise TypeError(msg)
            values = [kwargs.get(name, 0) for name in cls._fields]

        for i, field in enumerate(cls._fields):
            match values[i]:
                case int():
                    values[i] = int(values[i])
                case float():
                    values[i] = float(values[i])
                case _:
                    msg = f"Invalid type for DimensionSet dimension '{field}': {values[i]!r}"
                    raise TypeError(msg)

        return super().__new__(cls, values)  # ty: ignore[invalid-argument-type]

    def __getnewargs__(
        self,
    ) -> tuple[
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
    ]:
        return tuple(self)

    @property
    def mass(self) -> int | float:
        """Power of the mass dimension."""
        return self[0]

    @property
    def length(self) -> int | float:
        """Power of the length dimension."""
        return self[1]

    @property
    def time(self) -> int | float:
        """Power of the time dimension."""
        return self[2]

    @property
    def temperature(self) -> int | float:
        """Power of the temperature dimension."""
        return self[3]

    @property
    def moles(self) -> int | float:
        """Power of the amount of substance dimension."""
        return self[4]

    @property
    def current(self) -> int | float:
        """Power of the electric current dimension."""
        return self[5]

    @property
    def luminous_intensity(self) -> int | float:
        """Power of the luminous intensity dimension."""
        return self[6]

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self, strict=True) if v != 0)})"

    @override
    def __add__(self, other: "DimensionSet", /) -> "DimensionSet":  # ty: ignore[invalid-method-override]
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot add DimensionSet with different dimensions: {self} + {other}"
            raise ValueError(msg)

        return self

    def __sub__(self, other: "DimensionSet", /) -> "DimensionSet":
        if not isinstance(other, DimensionSet):
            return NotImplemented

        if self != other:
            msg = f"Cannot subtract DimensionSet with different dimensions: {self} - {other}"
            raise ValueError(msg)

        return self

    @override
    def __mul__(self, other: "DimensionSet", /) -> "DimensionSet":  # ty: ignore[invalid-method-override]
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a + b for a, b in zip(self, other, strict=True)))

    def __truediv__(self, other: "DimensionSet", /) -> "DimensionSet":
        if not isinstance(other, DimensionSet):
            return NotImplemented

        return DimensionSet(*(a - b for a, b in zip(self, other, strict=True)))

    def __pow__(self, exponent: float, /) -> "DimensionSet":
        if not isinstance(exponent, (int, float)):
            return NotImplemented

        return DimensionSet(*(a * exponent for a in self))

    def __bool__(self) -> bool:
        return any(v != 0 for v in self)


_NAMED_DIMENSIONS: dict[str, DimensionSet] = {}
_NAMED_DIMENSION_IDS: dict[int, str] = {}


def _register_named_dimension(name: str, dimensions: DimensionSet) -> DimensionSet:
    assert name not in _NAMED_DIMENSIONS
    if (id_ := id(dimensions)) in _NAMED_DIMENSION_IDS:
        dimensions = DimensionSet(*dimensions)
        id_ = id(dimensions)
    assert id_ not in _NAMED_DIMENSION_IDS
    _NAMED_DIMENSIONS[name] = dimensions
    _NAMED_DIMENSION_IDS[id_] = name
    return dimensions


# https://github.com/OpenFOAM/OpenFOAM-14/blob/master/src/OpenFOAM/dimensionSet/dimensions.C
_register_named_dimension("dimless", DimensionSet())
_register_named_dimension("mass", DimensionSet(mass=1))
_register_named_dimension("length", DimensionSet(length=1))
_register_named_dimension("time", DimensionSet(time=1))
_register_named_dimension("temperature", DimensionSet(temperature=1))
_register_named_dimension("moles", DimensionSet(moles=1))
_register_named_dimension("current", DimensionSet(current=1))
_register_named_dimension("luminousIntensity", DimensionSet(luminous_intensity=1))
_register_named_dimension("area", _NAMED_DIMENSIONS["length"] ** 2)
_register_named_dimension("volume", _NAMED_DIMENSIONS["length"] ** 3)
_register_named_dimension(
    "rate", _NAMED_DIMENSIONS["dimless"] / _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "velocity", _NAMED_DIMENSIONS["length"] / _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "momentum", _NAMED_DIMENSIONS["mass"] * _NAMED_DIMENSIONS["velocity"]
)
_register_named_dimension(
    "acceleration", _NAMED_DIMENSIONS["velocity"] / _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "density", _NAMED_DIMENSIONS["mass"] / _NAMED_DIMENSIONS["volume"]
)
_register_named_dimension(
    "momentumDensity", _NAMED_DIMENSIONS["momentum"] / _NAMED_DIMENSIONS["volume"]
)
_register_named_dimension(
    "force", _NAMED_DIMENSIONS["mass"] * _NAMED_DIMENSIONS["acceleration"]
)
_register_named_dimension(
    "energy", _NAMED_DIMENSIONS["force"] * _NAMED_DIMENSIONS["length"]
)
_register_named_dimension(
    "energyDensity", _NAMED_DIMENSIONS["energy"] / _NAMED_DIMENSIONS["volume"]
)
_register_named_dimension(
    "specificEnergy", _NAMED_DIMENSIONS["energy"] / _NAMED_DIMENSIONS["mass"]
)
_register_named_dimension("kineticEnergy", _NAMED_DIMENSIONS["energy"])
_register_named_dimension(
    "kineticEnergyDensity",
    _NAMED_DIMENSIONS["kineticEnergy"] / _NAMED_DIMENSIONS["volume"],
)
_register_named_dimension(
    "power", _NAMED_DIMENSIONS["energy"] / _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "powerDensity", _NAMED_DIMENSIONS["power"] / _NAMED_DIMENSIONS["volume"]
)
_register_named_dimension(
    "specificPower", _NAMED_DIMENSIONS["power"] / _NAMED_DIMENSIONS["mass"]
)
_register_named_dimension(
    "entropy", _NAMED_DIMENSIONS["energy"] / _NAMED_DIMENSIONS["temperature"]
)
_register_named_dimension(
    "specificEntropy", _NAMED_DIMENSIONS["entropy"] / _NAMED_DIMENSIONS["mass"]
)
_register_named_dimension(
    "heatCapacity", _NAMED_DIMENSIONS["energy"] / _NAMED_DIMENSIONS["temperature"]
)
_register_named_dimension(
    "specificHeatCapacity",
    _NAMED_DIMENSIONS["heatCapacity"] / _NAMED_DIMENSIONS["mass"],
)
_register_named_dimension("gasConstant", _NAMED_DIMENSIONS["specificHeatCapacity"])
_register_named_dimension(
    "pressure", _NAMED_DIMENSIONS["force"] / _NAMED_DIMENSIONS["area"]
)
_register_named_dimension(
    "kinematicPressure", _NAMED_DIMENSIONS["pressure"] / _NAMED_DIMENSIONS["density"]
)
_register_named_dimension(
    "compressibility", _NAMED_DIMENSIONS["density"] / _NAMED_DIMENSIONS["pressure"]
)
_register_named_dimension(
    "kinematicViscosity", _NAMED_DIMENSIONS["area"] / _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "dynamicViscosity",
    _NAMED_DIMENSIONS["density"] * _NAMED_DIMENSIONS["kinematicViscosity"],
)
_register_named_dimension(
    "kinematicDiffusivity", _NAMED_DIMENSIONS["kinematicViscosity"]
)
_register_named_dimension("dynamicDiffusivity", _NAMED_DIMENSIONS["dynamicViscosity"])
_register_named_dimension(
    "thermalConductivity",
    _NAMED_DIMENSIONS["power"]
    / _NAMED_DIMENSIONS["length"]
    / _NAMED_DIMENSIONS["temperature"],
)
_register_named_dimension("turbulentKineticEnergy", _NAMED_DIMENSIONS["velocity"] ** 2)
_register_named_dimension(
    "kinematicStress", _NAMED_DIMENSIONS["turbulentKineticEnergy"]
)
_register_named_dimension("ReynoldsStress", _NAMED_DIMENSIONS["kinematicStress"])
_register_named_dimension(
    "turbulentEpsilon",
    _NAMED_DIMENSIONS["turbulentKineticEnergy"] / _NAMED_DIMENSIONS["time"],
)
_register_named_dimension("turbulentOmega", _NAMED_DIMENSIONS["rate"])
_register_named_dimension("turbulentViscosity", _NAMED_DIMENSIONS["kinematicViscosity"])
_register_named_dimension(
    "volumetricFlux",
    _NAMED_DIMENSIONS["area"] * _NAMED_DIMENSIONS["velocity"],
)
_register_named_dimension(
    "volumetricFluxDensity",
    _NAMED_DIMENSIONS["volumetricFlux"] / _NAMED_DIMENSIONS["area"],
)
_register_named_dimension(
    "massFlux",
    _NAMED_DIMENSIONS["density"] * _NAMED_DIMENSIONS["volumetricFlux"],
)
_register_named_dimension(
    "massFluxDensity",
    _NAMED_DIMENSIONS["massFlux"] / _NAMED_DIMENSIONS["area"],
)
_register_named_dimension("heatFlux", _NAMED_DIMENSIONS["power"])
_register_named_dimension(
    "heatFluxDensity",
    _NAMED_DIMENSIONS["heatFlux"] / _NAMED_DIMENSIONS["area"],
)
_register_named_dimension(
    "charge", _NAMED_DIMENSIONS["current"] * _NAMED_DIMENSIONS["time"]
)
_register_named_dimension(
    "chargeDensity", _NAMED_DIMENSIONS["charge"] / _NAMED_DIMENSIONS["volume"]
)
_register_named_dimension(
    "electricPotential", _NAMED_DIMENSIONS["power"] / _NAMED_DIMENSIONS["current"]
)
_register_named_dimension(
    "magneticFluxDensity",
    _NAMED_DIMENSIONS["force"]
    / (_NAMED_DIMENSIONS["length"] * _NAMED_DIMENSIONS["current"]),
)
_register_named_dimension(
    "magneticFluxPressure",
    _NAMED_DIMENSIONS["magneticFluxDensity"] * _NAMED_DIMENSIONS["velocity"],
)


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
        value: "TensorLike",
        dimensions: "DimensionSetLike",
        name: str | None = None,
    ) -> None:
        from ..typing import Tensor  # noqa: PLC0415
        from ._normalization import normalized  # noqa: PLC0415

        if isinstance(value, np.ndarray):
            value = value.copy()
        self.value: Tensor = normalized(value, target=Tensor)  # ty: ignore[no-matching-overload]

        if not isinstance(dimensions, DimensionSet):
            self.dimensions = DimensionSet(*dimensions)
        else:
            self.dimensions = dimensions

        if name is not None and name != normalized(name, target=str):
            msg = f"Invalid name for Dimensioned: {name!r}"
            raise ValueError(msg)
        self.name = name

    @override
    def __repr__(self) -> str:
        if self.name is not None:
            return (
                f"{type(self).__name__}({self.value}, {self.dimensions}, {self.name})"
            )
        return f"{type(self).__name__}({self.value}, {self.dimensions})"

    def __add__(self, other: "Dimensioned | Tensor", /) -> "Dimensioned":
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value + other.value,
            self.dimensions + other.dimensions,
            f"{self.name}+{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __sub__(self, other: "Dimensioned | Tensor", /) -> "Dimensioned":
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value - other.value,
            self.dimensions - other.dimensions,
            f"{self.name}-{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __mul__(self, other: "Dimensioned | Tensor", /) -> "Dimensioned":
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value * other.value,
            self.dimensions * other.dimensions,
            f"{self.name}*{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __truediv__(self, other: "Dimensioned | Tensor", /) -> "Dimensioned":
        if not isinstance(other, Dimensioned):
            other = Dimensioned(other, DimensionSet())

        return Dimensioned(
            self.value / other.value,
            self.dimensions / other.dimensions,
            f"{self.name}/{other.name}"
            if self.name is not None and other.name is not None
            else None,
        )

    def __pow__(self, exponent: float, /) -> "Dimensioned":
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

    @overload
    def __array__(
        self, dtype: None = ..., *, copy: bool | None = ...
    ) -> np.ndarray[tuple[()] | tuple[Literal[3, 6, 9]], np.dtype[np.float64]]: ...

    @overload
    def __array__(
        self, dtype: np.dtype[_T], *, copy: bool | None = ...
    ) -> np.ndarray[tuple[()] | tuple[Literal[3, 6, 9]], np.dtype[_T]]: ...

    def __array__(
        self, dtype: np.dtype | None = None, *, copy: bool | None = None
    ) -> np.ndarray[tuple[()] | tuple[Literal[3, 6, 9]], np.dtype[np.float64 | _T]]:
        if self.dimensions:
            msg = f"Cannot convert non-dimensionless Dimensioned object to array: {self.dimensions}"
            raise ValueError(msg)
        return np.array(self.value, dtype=dtype, copy=copy)
