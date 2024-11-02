import os
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator
else:
    from typing import AsyncGenerator

import pytest
import pytest_asyncio
from foamlib import AsyncFoamCase, AsyncSlurmFoamCase, CalledProcessError


@pytest_asyncio.fixture(params=[AsyncFoamCase, AsyncSlurmFoamCase])
async def flange(request: pytest.FixtureRequest) -> AsyncGenerator[AsyncFoamCase, None]:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "basic" / "laplacianFoam" / "flange"
    of11_path = tutorials_path / "legacy" / "basic" / "laplacianFoam" / "flange"

    case = request.param(path if path.exists() else of11_path)
    assert isinstance(case, AsyncFoamCase)

    async with case.clone() as clone:
        yield clone


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [True, False])
async def test_run(flange: AsyncFoamCase, *, parallel: bool) -> None:
    if parallel:
        if not (flange.path / "Allrun-parallel").exists():
            pytest.skip()
        with flange.decompose_par_dict as d:
            assert d["method"] == "scotch"
            d["numberOfSubdomains"] = 2

    if isinstance(flange, AsyncSlurmFoamCase):
        await flange.run(parallel=parallel, fallback=True)
    else:
        await flange.run(parallel=parallel)

    if parallel:
        await flange.reconstruct_par()

    await flange.clean()

    if isinstance(flange, AsyncSlurmFoamCase):
        await flange.run(parallel=parallel, fallback=True)
    else:
        await flange.run(parallel=parallel)


@pytest.mark.asyncio
async def test_run_cmd(flange: AsyncFoamCase) -> None:
    if not flange:
        await flange.restore_0_dir()

    ans_path = (
        Path(os.environ["FOAM_TUTORIALS"]) / "resources" / "geometry" / "flange.ans"
    )
    if not ans_path.exists():
        ans_path = Path("flange.ans")

    await flange.run(
        [
            "ansysToFoam",
            ans_path,
            "-scale",
            "0.001",
        ],
        cpus=0,
    )

    if isinstance(flange, AsyncSlurmFoamCase):
        await flange.run([flange.application], fallback=True)
    else:
        await flange.run([flange.application])


@pytest.mark.asyncio
async def test_run_cmd_shell(flange: AsyncFoamCase) -> None:
    if not flange:
        await flange.restore_0_dir()

    try:
        await flange.run(
            'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001',
            cpus=0,
        )
    except CalledProcessError:
        await flange.run('ansysToFoam "flange.ans" -scale 0.001', cpus=0)

    if isinstance(flange, AsyncSlurmFoamCase):
        await flange.run(flange.application, fallback=True)
    else:
        await flange.run(flange.application)


def test_path(flange: AsyncFoamCase) -> None:
    assert Path(flange) == flange.path
