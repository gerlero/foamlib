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
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07633/status.svg)](https://doi.org/10.21105/joss.07633)


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

* With [Homebrew](https://brew.sh):

    ```bash
    brew install gerlero/openfoam/foamlib
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

## 📘 Documentation

For details on how to use **foamlib**, check out the [documentation](https://foamlib.readthedocs.io/).

## 🙋 Support

If you have any questions or need help, feel free to open a [discussion](https://github.com/gerlero/foamlib/discussions).

If you believe you have found a bug in **foamlib**, please open an [issue](https://github.com/gerlero/foamlib/issues).

## 🧑‍💻 Contributing

You're welcome to contribute to **foamlib**! Check out the [contributing guidelines](CONTRIBUTING.md) for more information.

## 🖋️Citation

If you find **foamlib** useful for your work, don't forget to cite it!

Citations help us a lot. You may find the following snippets useful:

<details>
<summary>BibTeX</summary>

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

<details>
<summary>APA</summary>

Gerlero, G. S., & Kler, P. A. (2025). foamlib: A modern Python package for working with OpenFOAM. Journal of Open Source Software, 10(109), 7633. https://doi.org/10.21105/joss.07633

</details>

## 👟 Footnotes

<a id="benchmark">[1]</a> foamlib 0.8.1 vs PyFoam 2023.7 on a MacBook Air (2020, M1) with 8 GB of RAM. [Benchmark script](benchmark/benchmark.py).
