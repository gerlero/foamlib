import os
import sys
from pathlib import Path
from typing import Sequence

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator
else:
    from typing import AsyncGenerator

import pytest
import pytest_asyncio
from foamlib import AsyncFoamCase


@pytest_asyncio.fixture
async def pitz() -> "AsyncGenerator[AsyncFoamCase]":
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "incompressible" / "simpleFoam" / "pitzDaily"
    of11_path = tutorials_path / "incompressibleFluid" / "pitzDaily"

    case = AsyncFoamCase(path if path.exists() else of11_path)

    async with case.clone() as clone:
        yield clone


@pytest.mark.asyncio
async def test_run(pitz: AsyncFoamCase) -> None:
    await pitz.run()
    await pitz.clean()
    await pitz.run()
    assert len(pitz) > 0
    internal = pitz[-1]["U"].internal_field
    assert isinstance(internal, Sequence)
    assert len(internal) == 12225


@pytest.mark.asyncio
async def test_double_clean(pitz: AsyncFoamCase) -> None:
    await pitz.clean()
    await pitz.clean(check=True)
    await pitz.run()
