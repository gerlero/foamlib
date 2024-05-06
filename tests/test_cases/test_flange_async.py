import os
from pathlib import Path

import pytest
import pytest_asyncio
from foamlib import AsyncFoamCase, CalledProcessError


@pytest_asyncio.fixture
async def flange(tmp_path: Path) -> AsyncFoamCase:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "basic" / "laplacianFoam" / "flange"
    of11_path = tutorials_path / "legacy" / "basic" / "laplacianFoam" / "flange"

    case = AsyncFoamCase(path if path.exists() else of11_path)

    return await case.clone(tmp_path / case.name)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [True, False])
async def test_run(flange: AsyncFoamCase, parallel: bool) -> None:
    if parallel and not (flange.path / "Allrun-parallel").exists():
        pytest.skip()
    await flange.run(parallel=parallel)
    if parallel:
        await flange.reconstruct_par()
    await flange.clean()
    await flange.run(parallel=parallel)


@pytest.mark.asyncio
async def test_run_cmd(flange: AsyncFoamCase) -> None:
    if (flange.path / "0.orig").exists():
        (flange.path / "0.orig").rename(flange.path / "0")

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
    )
    await flange.run(script=False)


@pytest.mark.asyncio
async def test_run_cmd_shell(flange: AsyncFoamCase) -> None:
    await flange.run("mv 0.orig 0", check=False)
    try:
        await flange.run(
            'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001'
        )
    except CalledProcessError:
        await flange.run('ansysToFoam "flange.ans" -scale 0.001')
    await flange.run(flange.application)


def test_path(flange: AsyncFoamCase) -> None:
    assert Path(flange) == flange.path
