import pytest

import os
from pathlib import Path

from foamlib import FoamCase

FLANGE = FoamCase(
    Path(os.environ["FOAM_TUTORIALS"]) / "basic" / "laplacianFoam" / "flange"
)


@pytest.fixture
def flange(tmp_path: Path) -> FoamCase:
    return FLANGE.clone(tmp_path / FLANGE.name)


@pytest.mark.parametrize("parallel", [True, False])
def test_run(flange: FoamCase, parallel: bool) -> None:
    flange.run(parallel=parallel)
    if parallel:
        flange.reconstruct_par()
    flange.clean()
    flange.run(parallel=parallel)


def test_run_cmd(flange: FoamCase) -> None:
    (flange.path / "0.orig").rename(flange.path / "0")
    flange.run(
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
    flange.run(script=False)
    flange.reconstruct_par()


def test_run_cmd_shell(flange: FoamCase) -> None:
    flange.run("mv 0.orig 0")
    flange.run(
        'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001'
    )
    flange.run("decomposePar")
    flange.run(flange.application, parallel=True)
    flange.run("reconstructPar")


def test_run_no_parallel(flange: FoamCase) -> None:
    with pytest.raises(ValueError):
        flange.run()


def test_path() -> None:
    assert Path(FLANGE) == FLANGE.path
