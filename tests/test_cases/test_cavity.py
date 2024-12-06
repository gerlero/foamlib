import os
import stat
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator

import numpy as np
import pytest
from foamlib import FoamCase


@pytest.fixture(params=[False, True])
def cavity(request: pytest.FixtureRequest) -> Generator[FoamCase, None, None]:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "incompressible" / "icoFoam" / "cavity" / "cavity"
    of11_path = tutorials_path / "incompressibleFluid" / "cavity"

    case = FoamCase(path if path.exists() else of11_path)

    with case.clone() as clone:
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


def test_run(cavity: FoamCase) -> None:
    cavity.run(parallel=False)
    cavity.clean()
    cavity.run(parallel=False)
    assert len(cavity) > 0
    internal = cavity[-1]["U"].internal_field
    assert isinstance(internal, np.ndarray)
    assert len(internal) == 400


def test_double_clean(cavity: FoamCase) -> None:
    cavity.clean()
    cavity.clean(check=True)
    cavity.run(parallel=False)


def test_cell_centers(cavity: FoamCase) -> None:
    cavity.block_mesh()
    C = cavity[0].cell_centers()
    assert isinstance(C.internal_field, np.ndarray)
    assert len(C.internal_field) == 400
