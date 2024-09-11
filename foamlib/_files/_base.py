import sys
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple, Union

if sys.version_info >= (3, 9):
    from collections.abc import Mapping, Sequence
else:
    from typing import Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    pass


class FoamFileBase:
    class DimensionSet(NamedTuple):
        mass: Union[int, float] = 0
        length: Union[int, float] = 0
        time: Union[int, float] = 0
        temperature: Union[int, float] = 0
        moles: Union[int, float] = 0
        current: Union[int, float] = 0
        luminous_intensity: Union[int, float] = 0

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({', '.join(f'{n}={v}' for n, v in zip(self._fields, self) if v != 0)})"

    @dataclass
    class Dimensioned:
        value: Union[int, float, Sequence[Union[int, float]]] = 0
        dimensions: Union["FoamFileBase.DimensionSet", Sequence[Union[int, float]]] = ()
        name: Optional[str] = None

        def __post_init__(self) -> None:
            if not isinstance(self.dimensions, FoamFileBase.DimensionSet):
                self.dimensions = FoamFileBase.DimensionSet(*self.dimensions)

    Data = Union[
        str,
        int,
        float,
        bool,
        Dimensioned,
        DimensionSet,
        Sequence["Data"],
        Mapping[str, "Data"],
    ]
    """
    A value that can be stored in an OpenFOAM file.
    """

    _Dict = Dict[str, Union["Data", "_Dict"]]
    _File = Dict[Optional[str], Union["Data", "_Dict"]]

    _SetData = Union[
        str,
        int,
        float,
        bool,
        Dimensioned,
        DimensionSet,
        Sequence["_SetData"],
        Mapping[str, "_SetData"],
        "np.ndarray[Tuple[()], np.dtype[np.generic]]",
        "np.ndarray[Tuple[int], np.dtype[np.generic]]",
        "np.ndarray[Tuple[int, int], np.dtype[np.generic]]",
    ]
