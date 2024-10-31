import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence


class FoamFileBase:
    class DimensionSet(NamedTuple):
        mass: float = 0
        length: float = 0
        time: float = 0
        temperature: float = 0
        moles: float = 0
        current: float = 0
        luminous_intensity: float = 0

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self) if v != 0)})"

    _Tensor = Union[
        float,
        Sequence[float],
        "np.ndarray[Tuple[()], np.dtype[np.generic]]",
        "np.ndarray[Tuple[int], np.dtype[np.generic]]",
    ]

    @dataclass
    class Dimensioned:
        value: "FoamFileBase._Tensor" = 0
        dimensions: Union["FoamFileBase.DimensionSet", Sequence[float]] = ()
        name: Optional[str] = None

        def __post_init__(self) -> None:
            if not isinstance(self.dimensions, FoamFileBase.DimensionSet):
                self.dimensions = FoamFileBase.DimensionSet(*self.dimensions)

    _Field = Union[
        _Tensor, Sequence[_Tensor], "np.ndarray[Tuple[int, int], np.dtype[np.generic]]"
    ]

    Data = Union[
        str,
        int,
        float,
        bool,
        Dimensioned,
        DimensionSet,
        Sequence["Data"],
        Mapping[str, "Data"],
        _Tensor,
        _Field,
    ]
    """
    A value that can be stored in an OpenFOAM file.
    """

    _Dict = Dict[str, Union["Data", "_Dict"]]
    _File = Dict[Optional[str], Union["Data", "_Dict"]]
