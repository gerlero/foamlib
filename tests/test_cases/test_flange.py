import os
import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import Generator
else:
    from typing import Generator

import pytest
from foamlib import CalledProcessError, FoamCase


@pytest.fixture
def flange() -> Generator[FoamCase, None, None]:
    tutorials_path = Path(os.environ["FOAM_TUTORIALS"])
    path = tutorials_path / "basic" / "laplacianFoam" / "flange"
    of11_path = tutorials_path / "legacy" / "basic" / "laplacianFoam" / "flange"

    case = FoamCase(path if path.exists() else of11_path)

    with case.clone() as clone:
        yield clone


@pytest.mark.parametrize("parallel", [True, False])
def test_run(flange: FoamCase, *, parallel: bool) -> None:
    if parallel:
        if not (flange.path / "Allrun-parallel").exists():
            pytest.skip()
        with flange.decompose_par_dict as d:
            assert d["method"] == "scotch"
            d["numberOfSubdomains"] = 2

    flange.run(parallel=parallel)
    if parallel:
        flange.reconstruct_par()
    flange.clean()
    flange.run(parallel=parallel)


def test_run_cmd(flange: FoamCase) -> None:
    if not flange:
        flange.restore_0_dir()

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
    flange.run([flange.application])


def test_run_cmd_shell(flange: FoamCase) -> None:
    if not flange:
        flange.restore_0_dir()

    try:
        flange.run(
            'ansysToFoam "$FOAM_TUTORIALS/resources/geometry/flange.ans" -scale 0.001'
        )
    except CalledProcessError:
        flange.run('ansysToFoam "flange.ans" -scale 0.001')

    flange.run(flange.application)


def test_path(flange: FoamCase) -> None:
    assert Path(flange) == flange.path
