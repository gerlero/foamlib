import os
import stat
import sys
from pathlib import Path
from typing import Sequence

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator
else:
    from typing import AsyncGenerator

import numpy as np
import pytest
import pytest_asyncio
from foamlib import AsyncFoamCase


@pytest_asyncio.fixture(params=[False, True])
async def cavity(request: pytest.FixtureRequest) -> AsyncGenerator[AsyncFoamCase, None]:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "incompressible" / "icoFoam" / "cavity" / "cavity"
    of11_path = tutorials_path / "incompressibleFluid" / "cavity"

    case = AsyncFoamCase(path if path.exists() else of11_path)

    async with case.clone() as clone:
        if request.param:
            run = clone.path / "run"
            assert not run.exists()
            assert not (clone.path / "Allrun").exists()
            run.write_text(
                "#!/usr/bin/env python3\nfrom pathlib import Path\nfrom foamlib import FoamCase\nFoamCase(Path(__file__).parent).run(parallel=False)"
            )
            run.chmod(run.stat().st_mode | stat.S_IEXEC)

            clean = clone.path / "clean"
            assert not clean.exists()
            assert not (clone.path / "Allclean").exists()
            clean.write_text(
                "#!/usr/bin/env python3\nfrom pathlib import Path\nfrom foamlib import FoamCase\nFoamCase(Path(__file__).parent).clean()"
            )
            clean.chmod(clean.stat().st_mode | stat.S_IEXEC)

        yield clone


@pytest.mark.asyncio
async def test_run(cavity: AsyncFoamCase) -> None:
    await cavity.run(parallel=False)
    await cavity.clean()
    await cavity.run(parallel=False)
    assert len(cavity) > 0
    internal = cavity[-1]["U"].internal_field
    assert isinstance(internal, np.ndarray)
    assert len(internal) == 400


@pytest.mark.asyncio
async def test_double_clean(cavity: AsyncFoamCase) -> None:
    await cavity.clean()
    await cavity.clean(check=True)
    await cavity.run(parallel=False)


@pytest.mark.asyncio
async def test_cell_centers(cavity: AsyncFoamCase) -> None:
    await cavity.block_mesh()
    C = await cavity[0].cell_centers()
    assert isinstance(C.internal_field, np.ndarray)
    assert len(C.internal_field) == 400


def test_map(cavity: AsyncFoamCase) -> None:
    async def f(x: Sequence[float]) -> float:
        async with cavity.clone() as clone:
            clone[0]["U"].boundary_field["movingWall"].value = [x[0], 0, 0]
            await clone.run(parallel=False)
            U = clone[-1]["U"].boundary_field["movingWall"].value
            assert not isinstance(U, (int, float))
            assert len(U) == 3
            ret = U[0]
            assert isinstance(ret, (int, float))
            return ret

    assert AsyncFoamCase.map(f, [[1], [2]]) == [1, 2]
    assert AsyncFoamCase.map(f, [[3]]) == [3]
