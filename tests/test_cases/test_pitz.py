import os
from pathlib import Path
from typing import Sequence

import pytest
from foamlib import FoamCase


@pytest.fixture
def pitz(tmp_path: Path) -> FoamCase:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "incompressible" / "simpleFoam" / "pitzDaily"
    of11_path = tutorials_path / "incompressibleFluid" / "pitzDaily"

    case = FoamCase(path if path.exists() else of11_path)

    return case.clone(tmp_path / case.name)


def test_run(pitz: FoamCase) -> None:
    pitz.run()
    pitz.clean()
    pitz.run()
    assert len(pitz) > 0
    internal = pitz[-1]["U"].internal_field
    assert isinstance(internal, Sequence)
    assert len(internal) == 12225


def test_double_clean(pitz: FoamCase) -> None:
    pitz.clean()
    pitz.clean(check=True)
    pitz.run()
