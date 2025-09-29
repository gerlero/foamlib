[<img alt="foamlib" src="https://github.com/gerlero/foamlib/raw/main/logo.png" height="65">](https://github.com/gerlero/foamlib)

[![Documentation](https://img.shields.io/readthedocs/foamlib)](https://foamlib.readthedocs.io/)
[![CI](https://github.com/gerlero/foamlib/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/foamlib/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/foamlib)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/foamlib/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/foamlib)](https://pypi.org/project/foamlib/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/foamlib)](https://anaconda.org/conda-forge/foamlib)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/foamlib)](https://pypi.org/project/foamlib/)
![OpenFOAM](https://img.shields.io/badge/openfoam-.com%20|%20.org-informational)
[![Docker](https://github.com/gerlero/foamlib/actions/workflows/docker.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/docker.yml)
[![Docker image](https://img.shields.io/badge/docker%20image-microfluidica%2Ffoamlib-0085a0)](https://hub.docker.com/r/microfluidica/foamlib/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07633/status.svg)](https://doi.org/10.21105/joss.07633)


**foamlib** is a modern Python package that provides an elegant, streamlined interface for interacting with OpenFOAM. It's designed to make OpenFOAM-based workflows more accessible, reproducible, and precise for researchers and engineers.

<div align="center">
<img alt="benchmark" src="https://github.com/gerlero/foamlib/raw/main/benchmark/benchmark.png" height="250">

_Loading a_ volVectorField _with one million cells_<sup>[1](#benchmark)</sup>
</div>


## 👋 Introduction

**foamlib** is a Python package designed to simplify and streamline OpenFOAM workflows. It provides:

- **🗄️ Intuitive file handling**: Read and write OpenFOAM configuration and field files as if they were Python objects
- **⚡ High performance**: Standalone parser supporting both ASCII and binary formats with or without compression
- **🔄 Async operations**: Run multiple cases in parallel with full [`asyncio`](https://docs.python.org/3/library/asyncio.html) support
- **🎯 Type safety**: A fully typed API for a better development experience
- **⚙️ Workflow automation**: Reduce boilerplate code for pre/post-processing and simulation management
- **🧩 Fully compatible**: Works with OpenFOAM from both [openfoam.com](https://www.openfoam.com/) and [openfoam.org](https://www.openfoam.org/)
- And more!

Compared to [PyFoam](https://openfoamwiki.net/index.php/Contrib/PyFoam) and other similar tools like [fluidfoam](https://github.com/fluiddyn/fluidfoam), [fluidsimfoam](https://foss.heptapod.net/fluiddyn/fluidsimfoam), and [Ofpp](https://github.com/xu-xianghua/ofpp), **foamlib** offers significant advantages in performance, usability, and modern Python compatibility.

## 🧱 Core components

**foamlib** provides these key classes for different aspects of OpenFOAM workflow automation:

### 📄 File handling
* **[`FoamFile`](https://foamlib.readthedocs.io/en/stable/files.html#foamlib.FoamFile)** - Read and write OpenFOAM configuration files as if they were Python `dict`s
* **[`FoamFieldFile`](https://foamlib.readthedocs.io/en/stable/files.html#foamlib.FoamFieldFile)** - Handle field files with support for ASCII and binary formats (with or without compression)

### 📁 Case management  
* **[`FoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.FoamCase)** - Configure, run, and access results of OpenFOAM cases
* **[`AsyncFoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.AsyncFoamCase)** - Asynchronous version for running multiple cases concurrently
* **[`AsyncSlurmFoamCase`](https://foamlib.readthedocs.io/en/stable/cases.html#foamlib.AsyncSlurmFoamCase)** - Specialized for Slurm-based HPC clusters

## 📦 Installation

Choose your preferred installation method:

<table>
<tr>
  <td><strong>✨ <a href="https://pypi.org/project/foamlib/">pip</a></strong></td>
  <td><code>pip install foamlib</code></td>
</tr>
<tr>
  <td><strong>🐍 <a href="https://anaconda.org/conda-forge/foamlib">conda</a></strong></td>
  <td><code>conda install -c conda-forge foamlib</code></td>
</tr>
<tr>
  <td><strong>🍺 <a href="https://github.com/gerlero/homebrew-openfoam">Homebrew</a></strong></td>
  <td><code>brew install gerlero/openfoam/foamlib</code></td>
</tr>
<tr>
  <td><strong>🐳 <a href="https://hub.docker.com/r/microfluidica/foamlib/">Docker</a></strong></td>
  <td><code>docker pull microfluidica/foamlib</code></td>
</table>

## 🚀 Quick start

Here's a simple example to get you started:

```python
import os
from pathlib import Path
from foamlib import FoamCase

# Clone and run a case
my_case = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily").clone("myCase")
my_case.run()

# Access results
latest_time = my_case[-1]
pressure = latest_time["p"].internal_field
velocity = latest_time["U"].internal_field

print(f"Max pressure: {max(pressure)}")
print(f"Velocity at first cell: {velocity[0]}")

# Clean up
my_case.clean()
```

## 📚 More usage examples

### 🐑 Clone a case

```python
import os
from pathlib import Path
from foamlib import FoamCase

pitz_tutorial = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")
my_pitz = pitz_tutorial.clone("myPitz")
```

### 🏃 Run the case and access results

```python
# Run the simulation
my_pitz.run()

# Access the latest time step
latest_time = my_pitz[-1]
p = latest_time["p"]
U = latest_time["U"]

print(f"Pressure field: {p.internal_field}")
print(f"Velocity field: {U.internal_field}")
```

### 🧹 Clean up and modify settings

```python
# Clean the case
my_pitz.clean()

# Modify control settings
my_pitz.control_dict["writeInterval"] = 10
my_pitz.control_dict["endTime"] = 2000
```

### 📝 Batch file modifications

```python
# Make multiple file changes efficiently
with my_pitz.fv_schemes as f:
    f["gradSchemes"]["default"] = f["divSchemes"]["default"]
    f["snGradSchemes"]["default"] = "uncorrected"
```

### ⏳ Asynchronous execution

```python
import asyncio
from foamlib import AsyncFoamCase

async def run_multiple_cases():
    """Run multiple cases concurrently."""
    base_case = AsyncFoamCase(my_pitz)
    
    # Create and run multiple cases with different parameters
    tasks = []
    for i, velocity in enumerate([1, 2, 3]):
        case = base_case.clone(f"case_{i}")
        case[0]["U"].boundary_field["inlet"].value = [velocity, 0, 0]
        tasks.append(case.run())
    
    # Wait for all cases to complete
    await asyncio.gather(*tasks)

# Run the async function
asyncio.run(run_multiple_cases())
```

### 🔢 Direct field file access

```python
import numpy as np
from foamlib import FoamFieldFile

# Read field data directly
U = FoamFieldFile("0/U")
print(f"Velocity field shape: {np.shape(U.internal_field)}")
print(f"Boundaries: {list(U.boundary_field)}")
```

### 🎯 Optimization with HPC clusters

```python
import os
from pathlib import Path
from foamlib import AsyncSlurmFoamCase
from scipy.optimize import differential_evolution

# Set up base case for optimization
base = AsyncSlurmFoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

async def objective_function(x):
    """Objective function for optimization."""
    async with base.clone() as case:
        # Set inlet velocity based on optimization parameters
        case[0]["U"].boundary_field["inlet"].value = [x[0], 0, 0]
        
        # Run with fallback to local execution if Slurm unavailable
        await case.run(fallback=True)
        
        # Return objective (minimize velocity magnitude at outlet)
        return abs(case[-1]["U"].internal_field[0][0])

# Run optimization with parallel jobs
result = differential_evolution(
    objective_function, 
    bounds=[(-1, 1)], 
    workers=AsyncSlurmFoamCase.map, 
    polish=False
)
print(f"Optimal inlet velocity: {result.x[0]}")
```

### 📄 Create Python-based `run`/`Allrun` scripts

```python
#!/usr/bin/env python3
"""Run the OpenFOAM case in this directory."""

from pathlib import Path
from foamlib import FoamCase

# Initialize case from this directory
case = FoamCase(Path(__file__).parent)

# Adjust simulation parameters
case.control_dict["endTime"] = 1000
case.control_dict["writeInterval"] = 100

# Run the simulation
print("Starting OpenFOAM simulation...")
case.run()
print("Simulation completed successfully!")
```

## 📘 Documentation

For more details on how to use **foamlib**, check out the [documentation](https://foamlib.readthedocs.io/).

## 🙋 Support

If you have any questions or need help, feel free to open a [discussion](https://github.com/gerlero/foamlib/discussions).

If you believe you have found a bug in **foamlib**, please open an [issue](https://github.com/gerlero/foamlib/issues).

## 🧑‍💻 Contributing

You're welcome to contribute to **foamlib**! Check out the [contributing guidelines](CONTRIBUTING.md) for more information.

## 🖋️ Citation

**foamlib** has been published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.07633)! 

If you use **foamlib** in your research, please cite our paper:

> Gerlero, G. S., & Kler, P. A. (2025). foamlib: A modern Python package for working with OpenFOAM. *Journal of Open Source Software*, 10(109), 7633. https://doi.org/10.21105/joss.07633

<details>
<summary>📋 BibTeX</summary>

```bibtex
@article{foamlib,
    author = {Gerlero, Gabriel S. and Kler, Pablo A.},
    doi = {10.21105/joss.07633},
    journal = {Journal of Open Source Software},
    month = may,
    number = {109},
    pages = {7633},
    title = {{foamlib: A modern Python package for working with OpenFOAM}},
    url = {https://joss.theoj.org/papers/10.21105/joss.07633},
    volume = {10},
    year = {2025}
}
```

</details>

## 👟 Footnotes

<a id="benchmark">[1]</a> foamlib 1.3.11 vs. PyFoam 2023.7 (Python 3.11.13) on an M3 MacBook Air. [Benchmark script](https://github.com/gerlero/foamlib/blob/main/benchmark/benchmark.py).
