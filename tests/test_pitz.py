import pytest

import os
from pathlib import Path
from typing import Sequence

from foamlib import FoamCase

PITZ = FoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "incompressible" / "simpleFoam" / "pitzDaily"
)


@pytest.fixture
def pitz(tmp_path: Path) -> FoamCase:
    return PITZ.clone(tmp_path / PITZ.name)


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


def test_run_parallel(pitz: FoamCase) -> None:
    with pytest.raises(RuntimeError):
        pitz.run(parallel=True)
