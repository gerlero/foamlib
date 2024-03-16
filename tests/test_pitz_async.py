import pytest
import pytest_asyncio

import os
from pathlib import Path
from typing import Sequence

from foamlib import AsyncFoamCase

PITZ = AsyncFoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "incompressible" / "simpleFoam" / "pitzDaily"
)


@pytest_asyncio.fixture
async def pitz(tmp_path: Path) -> AsyncFoamCase:
    return await PITZ.clone(tmp_path / PITZ.name)


@pytest.mark.asyncio
async def test_run(pitz: AsyncFoamCase) -> None:
    await pitz.run()
    await pitz.clean()
    await pitz.run()
    assert len(pitz) > 0
    internal = pitz[-1]["U"].internal_field
    assert isinstance(internal, Sequence)
    assert len(internal) > 0
    assert len(internal) == 12225


@pytest.mark.asyncio
async def test_double_clean(pitz: AsyncFoamCase) -> None:
    await pitz.clean()
    await pitz.clean(check=True)
    await pitz.run()


@pytest.mark.asyncio
async def test_run_parallel(pitz: AsyncFoamCase) -> None:
    with pytest.raises(RuntimeError):
        await pitz.run(parallel=True)
