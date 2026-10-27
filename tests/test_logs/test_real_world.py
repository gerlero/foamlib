"""Wordings that only real logs revealed. Every case here was a live bug.

The corpus these came from is 364 logs (23.9 MB) from 104 public repositories;
it is not redistributed, so each finding is pinned here as a minimal log.
"""

from __future__ import annotations

import pytest

from foamlib.logs import FoamLog, Kind, SolverSource

BANNER_ORG = (
    "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
    "|  \\    /   O peration     | Version:  4.0                                   |\n"
    "|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n"
)


def test_time_line_carries_a_unit_in_org_v11() -> None:
    """.org v11+ prints "Time = 1s"; a regex anchored on the number sees one step."""
    log = FoamLog.loads(
        "Build  : 12-abc\n"
        "Exec   : foamRun -solver incompressibleFluid\n"
        "nProcs : 1\n"
        "Time = 1s\n"
        "GAMG:  Solving for p, Initial residual = 1, "
        "Final residual = 1e-07, No Iterations 4\n"
        "Time = 2s\n"
        "GAMG:  Solving for p, Initial residual = 0.1, "
        "Final residual = 1e-08, No Iterations 3\n"
        "End\n"
    )
    assert log.n_steps == 2
    assert log.last_time == 2.0


@pytest.mark.parametrize(
    ("banner", "fork", "version"),
    [
        (BANNER_ORG, "org", "4.0"),
        (
            (
                "Using: OpenFOAM-2412 (2412) - visit www.openfoam.com\n"
                "Build: _2c4871ff-20250317\n"
            ),
            "com",
            "2412",
        ),
        (
            (
                "| RapidCFD by simFlow (sim-flow.com)                         |\n"
                "Build  : dev-f3775ac96129\n"
            ),
            "rapidcfd",
            "dev",
        ),
        (
            "| \\  /  A nd  | Website:  https://openfoam.org\nBuild  : 12-abc\n",
            "org",
            "12",
        ),
    ],
    ids=["web_not_website", "esi_using_line", "rapidcfd", "modern_org"],
)
def test_fork_and_version_from_real_banners(
    banner: str, fork: str, version: str
) -> None:
    log = FoamLog.loads(banner + "Exec   : icoFoam\nnProcs : 1\n")
    assert log.fork == fork
    assert log.version == version


def test_fork_survives_a_noisy_preamble() -> None:
    """A 64-rank run repeats its preamble per rank, and some builds emit
    hundreds of debug lines, so the domain can be far down the file."""
    log = FoamLog.loads(
        "mode\n" * 400 + "|   \\  /  A nd  | Website:  https://openfoam.org\n"
        "Build  : 12-abc\nExec   : simpleFoam\nnProcs : 1\n"
    )
    assert log.fork == "org"


@pytest.mark.parametrize(
    ("body", "reason", "detail_key", "detail_value"),
    [
        (
            (
                "--> FOAM FATAL IO ERROR:\n"
                "Entry 'Phi' not found in dictionary "
                '"/case/system/fvSolution/solvers"\n'
            ),
            "entry_not_found",
            "entry",
            "Phi",
        ),
        (
            (
                "--> FOAM FATAL ERROR: (openfoam-2506)\n"
                'Cannot find file "points" in directory "air/polyMesh" in times "0"\n'
            ),
            "missing_file",
            "path",
            "points",
        ),
        (
            (
                "--> FOAM FATAL IO ERROR: (openfoam-2206)\n"
                "Wrong token type - expected word, found on line 121: "
                "punctuation ')'\n"
            ),
            "dict_syntax_error",
            "line",
            "121",
        ),
        (
            (
                "mpirun was unable to find the specified executable file, and\n"
                "therefore did not launch the job.\n"
            ),
            "mpi_executable_not_found",
            None,
            None,
        ),
    ],
    ids=[
        "entry_not_found",
        "capitalised_cannot_find_file",
        "syntax_error",
        "no_binary",
    ],
)
def test_named_failures_from_the_corpus(
    body: str, reason: str, detail_key: str | None, detail_value: str | None
) -> None:
    log = FoamLog.loads("Build  : 2206\nExec   : potentialFoam\nnProcs : 1\n\n" + body)
    assert log.reason == reason
    if detail_key is not None:
        assert log.diagnostics[0].detail[detail_key] == detail_value
    assert log.fix["note"]


def test_launcher_failure_is_its_own_kind() -> None:
    log = FoamLog.loads(
        "mpirun has detected an attempt to run as root.\n", name="log.simpleFoam"
    )
    assert log.kind is Kind.LAUNCHER_FAILURE
    assert log.reason == "mpi_run_as_root"


def test_esi_temperature_format() -> None:
    """ESI reactingFoam writes "min/max(T) = 300, 1200" where cht writes
    "Min/max T:300 348.2"."""
    log = FoamLog.loads(
        "Build  : 2206\nExec   : reactingFoam\nnProcs : 1\n"
        "Courant Number mean: 0.1 max: 0.2\n"
        "Time = 1.1999e-06\n"
        "min/max(T) = 300, 1200.0305\n"
        "ExecutionTime = 0.06 s  ClockTime = 0 s\nEnd\n"
    )
    assert log.temperature_range == {"global": (300.0, 1200.0305)}


def test_blanked_header_falls_back_to_the_filename() -> None:
    """Logs exist with the header fields present but empty, and with no header
    at all. The caller is told the name was inferred."""
    log = FoamLog.loads(
        "Build  : 4.0\nExec   :\nCase   :\nnProcs : 1\n\nCreating block mesh from\n",
        name="log.blockMesh",
    )
    assert log.solver == "blockMesh"
    assert log.solver_source is SolverSource.FILENAME
    assert log.kind is Kind.UTILITY


def test_deprecated_solver_shim_is_reported() -> None:
    """In .org v11+ old solver names silently run a different module, and the
    wording of the notice varies."""
    log = FoamLog.loads(
        "Build  : 12-abc\nExec   : rhoSimpleFoam\nnProcs : 1\n\n"
        "rhoSimpleFoam has been superseded and replaced by the more general\n"
        "fluid solver module which will be executed using\n\n"
        "    foamRun -solver fluid\n\nSelecting solver fluid\n"
    )
    assert any(d.id == "solver_superseded" for d in log.diagnostics)


def test_foam_multi_run_is_not_named_after_a_region() -> None:
    """foamMultiRun selects one module per region; the last one is not the run."""
    log = FoamLog.loads(
        "Build  : 12-abc\nExec   : foamMultiRun\nnProcs : 1\n"
        "Selecting solver solid\nSelecting solver fluid\n"
    )
    assert log.solver == "foamMultiRun"
    assert list(log.solver_modules) == ["solid", "fluid"]


def test_converged_residual_starting_from_zero_is_not_divergence() -> None:
    """A field whose first final residual is exactly 0 made a pure ratio test
    call a converged 1e-9 "diverging". Two of three real interFoam logs hit it."""
    log = FoamLog.loads(
        "Build  : 2206\nExec   : interFoam\nnProcs : 1\n"
        "Courant Number mean: 0.1 max: 0.2\nTime = 1\n"
        "MULES:  Solving for alpha.water, Initial residual = 0, "
        "Final residual = 0, No Iterations 0\nTime = 2\n"
        "MULES:  Solving for alpha.water, Initial residual = 1e-09, "
        "Final residual = 1.58e-09, No Iterations 2\nEnd\n"
    )
    assert log.diverging == ()
    assert log.healthy
