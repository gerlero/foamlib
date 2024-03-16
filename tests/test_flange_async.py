import pytest
import pytest_asyncio

import os
from pathlib import Path

from foamlib import AsyncFoamCase

FLANGE = AsyncFoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "basic" / "laplacianFoam" / "flange"
)


@pytest_asyncio.fixture
async def flange(tmp_path: Path) -> AsyncFoamCase:
    return await FLANGE.clone(tmp_path / FLANGE.name)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [True, False])
async def test_run(flange: AsyncFoamCase, parallel: bool) -> None:
    await flange.run(parallel=parallel)
    if parallel:
        await flange.reconstruct_par()
    await flange.clean()
    await flange.run(parallel=parallel)


@pytest.mark.asyncio
async def test_run_cmd(flange: AsyncFoamCase) -> None:
    (flange.path / "0.orig").rename(flange.path / "0")
    await flange.run(
        [
            "ansysToFoam",
            Path(os.environ["FOAM_TUTORIALS"])
            / "resources"
            / "geometry"
            / "flange.ans",
            "-scale",
            "0.001",
        ],
    )
    await flange.run(script=False)
    await flange.reconstruct_par()


@pytest.mark.asyncio
async def test_run_cmd_shell(flange: AsyncFoamCase) -> None:
    await flange.run("mv 0.orig 0")
    await flange.run(
        'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001'
    )
    await flange.run("decomposePar")
    await flange.run(flange.application, parallel=True, cpus=4)
    await flange.run("reconstructPar")


@pytest.mark.asyncio
async def test_run_no_parallel(flange: AsyncFoamCase) -> None:
    with pytest.raises(ValueError):
        await flange.run()


def test_path() -> None:
    assert Path(FLANGE) == FLANGE.path
