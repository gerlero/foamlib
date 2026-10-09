import numpy as np


def expect_field(keywords: tuple[str, ...], /) -> bool:
    match keywords:
        case ("internalField",):
            return True
        case ("boundaryField", _, k) if k in ("value", "gradient") or k.endswith(
            ("Value", "Gradient")
        ):
            return True
    return False


def vol_field_class(field: object, /) -> str:
    match field:
        case np.ndarray(shape=(3,) | (_, 3), dtype=np.dtype(kind="f")):
            return "volVectorField"
        case np.ndarray(shape=(6,) | (_, 6), dtype=np.dtype(kind="f")):
            return "volSymmTensorField"
        case np.ndarray(shape=(9,) | (_, 9), dtype=np.dtype(kind="f")):
            return "volTensorField"
        case float() | np.ndarray(shape=(_,), dtype=np.dtype(kind="f")):
            return "volScalarField"
        case _:
            msg = "Cannot determine field class for data"
            raise TypeError(msg)
