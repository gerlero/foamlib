from pathlib import Path

import numpy as np
import pytest
from foamlib import FoamCase
from scipy.special import erfc


def test_example(tmp_path: Path) -> None:
    path = tmp_path / "diffusionCheck"
    path.mkdir()
    (path / "system").mkdir()
    (path / "constant").mkdir()
    (path / "0").mkdir()

    case = FoamCase(path)

    with case.control_dict as f:
        f["application"] = "scalarTransportFoam"
        f["startFrom"] = "latestTime"
        f["stopAt"] = "endTime"
        f["endTime"] = 5
        f["deltaT"] = 1e-3
        f["writeControl"] = "adjustableRunTime"
        f["writeInterval"] = 1
        f["purgeWrite"] = 0
        f["writeFormat"] = "ascii"
        f["writePrecision"] = 6
        f["writeCompression"] = False
        f["timeFormat"] = "general"
        f["timePrecision"] = 6
        f["adjustTimeStep"] = False
        f["runTimeModifiable"] = False

    with case.fv_schemes as f:
        f["ddtSchemes"] = {"default": "Euler"}
        f["gradSchemes"] = {"default": "Gauss linear"}
        f["divSchemes"] = {
            "default": "none",
            "div(phi,U)": "Gauss linear",
            "div(phi,T)": "Gauss linear",
        }
        f["laplacianSchemes"] = {"default": "Gauss linear corrected"}

    with case.fv_solution as f:
        f["solvers"] = {
            "T": {
                "solver": "PBiCG",
                "preconditioner": "DILU",
                "tolerance": 1e-6,
                "relTol": 0,
            }
        }
        f["SIMPLE"] = {}

    with case.block_mesh_dict as f:
        f["scale"] = 1
        f["vertices"] = [
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
        ]
        f["blocks"] = [
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
        ]
        f["edges"] = []
        boundary: list[tuple[str, dict[str, str | list[list[int]]]]] = [
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
        ]
        f["boundary"] = boundary
        f["mergePatchPairs"] = []

    with case.transport_properties as f:
        f["DT"] = f.Dimensioned(1e-3, f.DimensionSet(length=2, time=-1), "DT")

    with case[0]["U"] as f:
        f.dimensions = f.DimensionSet(length=1, time=-1)
        f.internal_field = [1, 0, 0]
        f.boundary_field = {
            "inletUp": {"type": "fixedValue", "value": [1, 0, 0]},
            "inletDown": {"type": "fixedValue", "value": [1, 0, 0]},
            "outletUp": {"type": "zeroGradient"},
            "outletDown": {"type": "zeroGradient"},
            "walls": {"type": "zeroGradient"},
            "frontAndBack": {"type": "empty"},
        }

    with case[0]["T"] as f:
        f.dimensions = f.DimensionSet(temperature=1)
        f.internal_field = 0
        f.boundary_field = {
            "inletUp": {"type": "fixedValue", "value": 0},
            "inletDown": {"type": "fixedValue", "value": 1},
            "outletUp": {"type": "zeroGradient"},
            "outletDown": {"type": "zeroGradient"},
            "walls": {"type": "zeroGradient"},
            "frontAndBack": {"type": "empty"},
        }

    case.run()

    x, y, z = case[0].cell_centers().internal_field.T  # type: ignore [union-attr]

    end = x == x.max()
    x = x[end]
    y = y[end]
    z = z[end]

    DT = case.transport_properties["DT"].value  # type: ignore [union-attr]
    assert isinstance(DT, float)

    U = case[0]["U"].internal_field[0]  # type: ignore [index]
    assert isinstance(U, (int, float))

    for time in case[1:]:
        if U * time.time < 2 * x.max():
            continue

        numerical = time["T"].internal_field[end]  # type: ignore [index]
        analytical = 0.5 * erfc((y - 0.5) / np.sqrt(4 * DT * x / U))
        assert numerical == pytest.approx(analytical, abs=1e-1)
