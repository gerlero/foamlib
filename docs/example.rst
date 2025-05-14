Example
=======

This example script sets up and runs a validation test case for the ``scalarTransportFoam`` solver, verifying the diffusion of a scalar field in a simplified 2D domain.

Overview
--------

- Creates a clean OpenFOAM case in the ``diffusionCheck`` subdirectory.
- Configures mesh geometry, solver settings, and initial/boundary conditions for scalar (``T``) and velocity (``U``) fields.
- Simulates a velocity-driven scalar transport where a temperature gradient is imposed across the inlet.
- Uses :class:`foamlib.FoamCase` and related utilities to manage OpenFOAM input files and execution.
- Computes an analytical solution using the complementary error function (:func:`scipy.special.erfc`) and compares it against numerical results.

Code
----

.. code-block:: python

    #!/usr/bin/env python3
    """Check the diffusion of a scalar field in a scalarTransportFoam case."""

    import shutil
    from pathlib import Path

    import numpy as np
    from scipy.special import erfc
    from foamlib import FoamCase, FoamFile

    path = Path(__file__).parent / "diffusionCheck"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
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
        f["divSchemes"] = {"default": "none", "div(phi,U)": "Gauss linear", "div(phi,T)": "Gauss linear"}
        f["laplacianSchemes"] = {"default": "Gauss linear corrected"}

    with case.fv_solution as f:
        f["solvers"] = {"T": {"solver": "PBiCG", "preconditioner": "DILU", "tolerance": 1e-6, "relTol": 0}}

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
            "hex", [0, 1, 2, 5, 6, 7, 8, 11], [400, 20, 1], "simpleGrading", [1, 1, 1],
            "hex", [5, 2, 3, 4, 11, 8, 9, 10], [400, 20, 1], "simpleGrading", [1, 1, 1],
        ]
        f["edges"] = []
        f["boundary"] = [
            ("inletUp", {"type": "patch", "faces": [[5, 4, 10, 11]]}),
            ("inletDown", {"type": "patch", "faces": [[0, 5, 11, 6]]}),
            ("outletUp", {"type": "patch", "faces": [[2, 3, 9, 8]]}),
            ("outletDown", {"type": "patch", "faces": [[1, 2, 8, 7]]}),
            ("walls", {"type": "wall", "faces": [[4, 3, 9, 10], [0, 1, 7, 6]]}),
            ("frontAndBack", {"type": "empty", "faces": [[0, 1, 2, 5], [5, 2, 3, 4], [6, 7, 8, 11], [11, 8, 9, 10]]}),
        ]
        f["mergePatchPairs"] = []

    with case.transport_properties as f:
        f["DT"] = FoamFile.Dimensioned(1e-3, f.DimensionSet(length=2, time=-1), "DT")

    with case[0]["U"] as f:
        f.dimensions = FoamFile.DimensionSet(length=1, time=-1)
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
        f.dimensions = FoamFile.DimensionSet(temperature=1)
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

    x, y, z = case[0].cell_centers().internal_field.T

    end = x == x.max()
    x = x[end]
    y = y[end]
    z = z[end]

    DT = case.transport_properties["DT"].value
    U = case[0]["U"].internal_field[0]

    for time in case[1:]:
        if U*time.time < 2*x.max():
            continue

        T = time["T"].internal_field[end]
        analytical = 0.5 * erfc((y - 0.5) / np.sqrt(4 * DT * x/U))
        if np.allclose(T, analytical, atol=0.1):
            print(f"Time {time.time}: OK")
        else:
            raise RuntimeError(f"Time {time.time}: {T} != {analytical}")
