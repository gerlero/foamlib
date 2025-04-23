[<img alt="foamlib" src="https://github.com/gerlero/foamlib/raw/main/logo.png" height="65">](https://github.com/gerlero/foamlib)

[![Documentation](https://img.shields.io/readthedocs/foamlib)](https://foamlib.readthedocs.io/)
[![CI](https://github.com/gerlero/foamlib/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/foamlib/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/foamlib)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/foamlib/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/foamlib)](https://pypi.org/project/foamlib/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/foamlib)](https://anaconda.org/conda-forge/foamlib)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/foamlib)](https://pypi.org/project/foamlib/)
![OpenFOAM](https://img.shields.io/badge/openfoam-.com%20|%20.org-informational)
[![Docker](https://github.com/gerlero/foamlib/actions/workflows/docker.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/docker.yml)
[![Docker image](https://img.shields.io/badge/docker%20image-microfluidica%2Ffoamlib-0085a0)](https://hub.docker.com/r/microfluidica/foamlib/)

**foamlib** provides a simple, modern, ergonomic and fast Python interface for interacting with [OpenFOAM](https://www.openfoam.com).

<div align="center">
<img alt="benchmark" src="https://github.com/gerlero/foamlib/raw/main/benchmark/benchmark.png" height="250">

Parsing a volVectorField with 200k cells.<sup>[1](#benchmark)</sup>
</div>


## 🚀 Introduction

**foamlib** is a Python package designed to simplify the manipulation of OpenFOAM cases and files. Its standalone parser makes it easy to work with OpenFOAM’s input/output files from Python, while its case-handling capabilities facilitate various execution workflows—reducing boilerplate code and enabling efficient Python-based pre- and post-processing, as well as simulation management.

Compared to [PyFoam](https://openfoamwiki.net/index.php/Contrib/PyFoam) and other similar tools like [fluidfoam](https://github.com/fluiddyn/fluidfoam), [fluidsimfoam](https://foss.heptapod.net/fluiddyn/fluidsimfoam), and [Ofpp](https://github.com/xu-xianghua/ofpp), **foamlib** offers advantages such as modern Python compatibility, support for binary-formatted fields, a fully type-hinted API, and asynchronous operations; making OpenFOAM workflows more accessible and streamlined.

## 👋 Basics

**foamlib** offers the following Python classes:

* [`FoamFile`](https://foamlib.readthedocs.io/en/stable/files.html#foamlib.FoamFile) (and [`FoamFieldFile`](https://foamlib.readthedocs.io/en/stable/files.html#foamlib.FoamFieldFile)): read-write access to OpenFOAM configuration and field files as if they were Python `dict`s, using `foamlib`'s own parser and in-place editor. Supports ASCII and binary field formats (with or without compression).
* [`FoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.FoamCase): a class for configuring, running, and accessing the results of OpenFOAM cases.
* [`AsyncFoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.AsyncFoamCase): variant of `FoamCase` with asynchronous methods for running multiple cases at once.
* [`AsyncSlurmFoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.AsyncSlurmFoamCase): subclass of `AsyncFoamCase` used for running cases on a Slurm cluster.

## ☑️ Get started

### 📦 Install

* With [pip](https://pypi.org/project/pip/):

    ```bash
    pip install foamlib
    ```

* With [conda](https://docs.conda.io/en/latest/):

    ```bash
    conda install -c conda-forge foamlib
    ```

### 🐑 Clone a case

```python
import os
from pathlib import Path
from foamlib import FoamCase

pitz_tutorial = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

my_pitz = pitz_tutorial.clone("myPitz")
```

### 🏃 Run the case

```python
my_pitz.run()
```

### 🔎 Access the results

```python
latest_time = my_pitz[-1]

p = latest_time["p"]
U = latest_time["U"]

print(p.internal_field)
print(U.internal_field)
```

### 🧹 Clean the case

```python
my_pitz.clean()
```

### ⚙️ Edit the `controlDict` file

```python
my_pitz.control_dict["writeInterval"] = 10
```

### 📝 Make multiple file reads and writes in a single go

```python
with my_pitz.fv_schemes as f:
    f["gradSchemes"]["default"] = f["divSchemes"]["default"]
    f["snGradSchemes"]["default"] = "uncorrected"
```

### ⏳ Run a case asynchronously

```python
import asyncio
from foamlib import AsyncFoamCase

async def run_case():
    my_pitz_async = AsyncFoamCase(my_pitz)
    await my_pitz_async.run()

asyncio.run(run_case())
```

### 🔢 Parse a field using the [`FoamFieldFile`](https://foamlib.readthedocs.io/en/stable/#foamlib.FoamFieldFile) class directly

```python
from foamlib import FoamFieldFile

U = FoamFieldFile(Path(my_pitz) / "0/U")

print(U.internal_field)
```

### 🔁 Run an optimization loop on a Slurm-based cluster

```python
import os
from pathlib import Path
from foamlib import AsyncSlurmFoamCase
from scipy.optimize import differential_evolution

base = AsyncSlurmFoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

async def cost(x):
    async with base.clone() as clone:
        clone[0]["U"].boundary_field["inlet"].value = [x[0], 0, 0]
        await clone.run(fallback=True) # Run locally if Slurm is not available
        return abs(clone[-1]["U"].internal_field[0][0])

result = differential_evolution(cost, bounds=[(-1, 1)], workers=AsyncSlurmFoamCase.map, polish=False)
```

### 📄 Use it to create a `run` (or `clean`) script
    
```python
#!/usr/bin/env python3
from pathlib import Path
from foamlib import FoamCase

case = FoamCase(Path(__file__).parent)
# Any additional configuration here
case.run()
```

## ▶️ A complete example

The following is a fully self-contained example that demonstrates how to create an OpenFOAM case from scratch, run it, and analyze the results.

<details>

<summary>Example</summary>

```python
#!/usr/bin/env python3
"""Check the diffusion of a scalar field in a scalarTransportFoam case."""

import shutil
from pathlib import Path

import numpy as np
from scipy.special import erfc
from foamlib import FoamCase

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
```

</details>


## 📘 API documentation

For more information on how to use **foamlibs**'s classes and methods, check out the [documentation](https://foamlib.readthedocs.io/).

## 🙋 Support

If you have any questions or need help, feel free to open a [discussion](https://github.com/gerlero/foamlib/discussions).

If you believe you have found a bug in **foamlib**, please open an [issue](https://github.com/gerlero/foamlib/issues).

## 🧑‍💻 Contributing

You're welcome to contribute to **foamlib**! Check out the [contributing guidelines](CONTRIBUTING.md) for more information.

## Footnotes

<a id="benchmark">[1]</a> foamlib 0.8.1 vs PyFoam 2023.7 on a MacBook Air (2020, M1) with 8 GB of RAM. [Benchmark script](benchmark/benchmark.py).
