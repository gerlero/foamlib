import os
from pathlib import Path

import pytest
from foamlib import CalledProcessError, FoamCase


@pytest.fixture
def flange(tmp_path: Path) -> FoamCase:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "basic" / "laplacianFoam" / "flange"
    of11_path = tutorials_path / "legacy" / "basic" / "laplacianFoam" / "flange"

    case = FoamCase(path if path.exists() else of11_path)

    return case.clone(tmp_path / case.name)


@pytest.mark.parametrize("parallel", [True, False])
def test_run(flange: FoamCase, parallel: bool) -> None:
    if parallel and not (flange.path / "Allrun-parallel").exists():
        pytest.skip()
    flange.run(parallel=parallel)
    if parallel:
        flange.reconstruct_par()
    flange.clean()
    flange.run(parallel=parallel)


def test_run_cmd(flange: FoamCase) -> None:
    if (flange.path / "0.orig").exists():
        (flange.path / "0.orig").rename(flange.path / "0")

    ans_path = (
        Path(os.environ["FOAM_TUTORIALS"]) / "resources" / "geometry" / "flange.ans"
    )
    if not ans_path.exists():
        ans_path = Path("flange.ans")

    flange.run(
        [
            "ansysToFoam",
            ans_path,
            "-scale",
            "0.001",
        ],
    )
    flange.run(script=False)


def test_run_cmd_shell(flange: FoamCase) -> None:
    flange.run("mv 0.orig 0", check=False)
    try:
        flange.run(
            'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001'
        )
    except CalledProcessError:
        flange.run('ansysToFoam "flange.ans" -scale 0.001')

    flange.run(flange.application, parallel=False)


def test_path(flange: FoamCase) -> None:
    assert Path(flange) == flange.path
