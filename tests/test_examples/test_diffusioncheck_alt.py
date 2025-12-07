from pathlib import Path

import numpy as np
import pytest
from foamlib import Dimensioned, DimensionSet, FoamCase
from scipy.special import erfc


def test_example(tmp_path: Path) -> None:
    path = tmp_path / "diffusionCheck"
    path.mkdir()
    (path / "system").mkdir()
    (path / "constant").mkdir()
    (path / "0").mkdir()

    case = FoamCase(path)

    case.control_dict[:] = {
        "application": "scalarTransportFoam",
        "startFrom": "latestTime",
        "stopAt": "endTime",
        "endTime": 5,
        "deltaT": 1e-3,
        "writeControl": "adjustableRunTime",
        "writeInterval": 1,
        "purgeWrite": 0,
        "writeFormat": "ascii",
        "writePrecision": 6,
        "writeCompression": False,
        "timeFormat": "general",
        "timePrecision": 6,
        "adjustTimeStep": False,
        "runTimeModifiable": False,
    }

    case.fv_schemes[:] = {
        "ddtSchemes": {"default": "Euler"},
        "gradSchemes": {"default": ("Gauss", "linear")},
        "divSchemes": {
            "default": "none",
            "div(phi,U)": ("Gauss", "linear"),
            "div(phi,T)": ("Gauss", "linear"),
        },
        "laplacianSchemes": {"default": ("Gauss", "linear", "corrected")},
    }

    case.fv_solution[:] = {
        "solvers": {
            "T": {
                "solver": "PBiCG",
                "preconditioner": "DILU",
                "tolerance": 1e-6,
                "relTol": 0,
            }
        },
        "SIMPLE": {},
    }

    case.block_mesh_dict[:] = {
        "scale": 1,
        "vertices": [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0.5, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0.5, 0],
            [0, 0, 0.1],
            [1, 0, 0.1],
            [1, 0.5, 0.1],
            [1, 1, 0.1],
            [0, 1, 0.1],
            [0, 0.5, 0.1],
        ],
        "blocks": [
            "hex",
            [0, 1, 2, 5, 6, 7, 8, 11],
            [400, 20, 1],
            "simpleGrading",
            [1, 1, 1],
            "hex",
            [5, 2, 3, 4, 11, 8, 9, 10],
            [400, 20, 1],
            "simpleGrading",
            [1, 1, 1],
        ],
        "edges": [],
        "boundary": [
            ("inletUp", {"type": "patch", "faces": [[5, 4, 10, 11]]}),
            ("inletDown", {"type": "patch", "faces": [[0, 5, 11, 6]]}),
            ("outletUp", {"type": "patch", "faces": [[2, 3, 9, 8]]}),
            ("outletDown", {"type": "patch", "faces": [[1, 2, 8, 7]]}),
            ("walls", {"type": "wall", "faces": [[4, 3, 9, 10], [0, 1, 7, 6]]}),
            (
                "frontAndBack",
                {
                    "type": "empty",
                    "faces": [
                        [0, 1, 2, 5],
                        [5, 2, 3, 4],
                        [6, 7, 8, 11],
                        [11, 8, 9, 10],
                    ],
                },
            ),
        ],
        "mergePatchPairs": [],
    }

    case.transport_properties["DT"] = Dimensioned(
        1e-3, DimensionSet(length=2, time=-1), "DT"
    )

    case[0]["U"][:] = {
        "dimensions": DimensionSet(length=1, time=-1),
        "internalField": [1, 0, 0],
        "boundaryField": {
            "inletUp": {"type": "fixedValue", "value": [1, 0, 0]},
            "inletDown": {"type": "fixedValue", "value": [1, 0, 0]},
            "outletUp": {"type": "zeroGradient"},
            "outletDown": {"type": "zeroGradient"},
            "walls": {"type": "zeroGradient"},
            "frontAndBack": {"type": "empty"},
        },
    }

    case[0]["T"][:] = {
        "dimensions": DimensionSet(temperature=1),
        "internalField": 0,
        "boundaryField": {
            "inletUp": {"type": "fixedValue", "value": 0},
            "inletDown": {"type": "fixedValue", "value": 1},
            "outletUp": {"type": "zeroGradient"},
            "outletDown": {"type": "zeroGradient"},
            "walls": {"type": "zeroGradient"},
            "frontAndBack": {"type": "empty"},
        },
    }

    case.run()

    internal_field = case[0].cell_centers().internal_field
    assert isinstance(internal_field, np.ndarray)
    x, y, z = internal_field.T

    end = x == x.max()
    x = x[end]
    y = y[end]
    z = z[end]

    DT = case.transport_properties["DT"].value  # ty: ignore[possibly-missing-attribute]
    assert isinstance(DT, float)

    internal_field = case[0]["U"].internal_field
    assert isinstance(internal_field, np.ndarray)
    U = internal_field[0]
    assert isinstance(U, (int, float))

    for time in case[1:]:
        if U * time.time < 2 * x.max():
            continue

        internal_field = time["T"].internal_field
        assert isinstance(internal_field, np.ndarray)
        numerical = internal_field[end]
        analytical = 0.5 * erfc((y - 0.5) / np.sqrt(4 * DT * x / U))
        assert numerical == pytest.approx(analytical, abs=1e-1)
