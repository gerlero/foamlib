# foamlib

[![Documentation](https://img.shields.io/readthedocs/foamlib)](https://foamlib.readthedocs.io/)
[![CI](https://github.com/gerlero/foamlib/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/foamlib/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/foamlib/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/foamlib)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/foamlib)](https://pypi.org/project/foamlib/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/foamlib)](https://pypi.org/project/foamlib/)


**foamlib** provides a simple, modern and ergonomic Python interface for interacting with [OpenFOAM](https://www.openfoam.com).

It offers the following classes (among a few others):

* [`FoamCase`](https://foamlib.readthedocs.io/en/stable/#foamlib.FoamCase): a class for manipulating, executing and accessing the results of OpenFOAM cases.
* [`AsyncFoamCase`](https://foamlib.readthedocs.io/en/stable/#foamlib.AsyncFoamCase): variant of `FoamCase` with asynchronous methods for running multiple cases at once.
* [`FoamFile`](https://foamlib.readthedocs.io/en/stable/#foamlib.FoamFile): read-write access to OpenFOAM configuration and field files as if they were Python `dict`s (uses OpenFOAM's `foamDictionary` utility with custom parsing).

## Get started

### Install

Install with [pip](https://pypi.org/project/pip/):

```bash
pip install foamlib
```

### Clone a case

```python
import os
from pathlib import Path
from foamlib import FoamCase

pitz_tutorial = FoamCase(Path(os.environ["FOAM_TUTORIALS"]) / "incompressible/simpleFoam/pitzDaily")

my_pitz = pitz_tutorial.clone("myPitz")
```

### Run the case

```python
my_pitz.run()
```

### Access the results

```python
latest_time = my_pitz[-1]

p = latest_time["p"]
U = latest_time["U"]

print(p.internal_field)
print(U.internal_field)
```

### Clean the case

```python
my_pitz.clean()
```

### Edit the `controlDict` file

```python
my_pitz.control_dict["writeInterval"] = 10
```

### Run a case asynchronously

```python
import asyncio
from foamlib import AsyncFoamCase

async def run_case():
    my_pitz_async = AsyncFoamCase(my_pitz)

    await my_pitz_async.run()

asyncio.run(run_case())
```

## Documentation

For more information, check out the [documentation](https://foamlib.readthedocs.io/).
