"""Behaviour on whole logs, one fixture per solver family.

Expected values are hand-derived from the fixtures, not captured from the
parser, so these tests can fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foamlib.logs import DIAGNOSTICS, FoamLog, Kind, SolverSource, Status, Trend

LOGS = Path(__file__).parent / "logs"


def test_transient_crash_after_a_courant_blow_up() -> None:
    log = FoamLog.load(LOGS / "log.pimpleFoam.diverged")

    assert log.solver == "pimpleFoam"
    assert log.exec_command == "pimpleFoam -parallel"
    assert log.fork == "org"
    assert log.version == "12"
    assert log.n_procs == 8
    assert not log.steady
    assert log.n_steps == 6
    assert log.last_time == pytest.approx(0.006)
    assert log.status is Status.FAILED
    assert log.reason == "fp_exception"
    assert log.courant_max == pytest.approx(41182.0)
    # the crashed step never printed an ExecutionTime of its own
    assert log.wallclock == pytest.approx(48.22)
    assert log.steps[-1].exec_time is None
    assert log.bounding == {"epsilon": 3, "k": 3}
    assert log.residuals["Ux"].trend is Trend.DIVERGING

    # fatals sort before warnings; the symptom must not outrank the cause
    assert [d.id for d in log.diagnostics] == ["fp_exception", "mpi_abort", "bounding"]
    # the specific diagnosis overrides the generic advice
    assert log.fix["adjustTimeStep"] is True
    assert "Courant" in log.fix["note"]
    assert not log.healthy


def test_steady_run_meeting_its_criteria() -> None:
    log = FoamLog.load(LOGS / "log.simpleFoam.converged")

    # a classic solver is its own executable
    assert log.solver_source is SolverSource.EXEC

    assert log.steady
    assert log.courant_max is None
    assert log.status is Status.CONVERGED
    assert log.converged_in == 247
    assert log.residuals["p"].trend is Trend.CONVERGING
    assert log.diagnostics == ()
    assert log.fix == {}
    assert log.healthy


def test_esi_multiphase_tracks_mass_conservation() -> None:
    log = FoamLog.load(LOGS / "log.interFoam.esi")

    assert log.fork == "com"
    assert log.version == "v2412"
    assert log.solver == "interFoam"
    assert log.interface_courant_max == pytest.approx(0.542)
    # MULES lines without residuals mean alpha.water is not a residual field
    assert set(log.fields) == {"Ux", "Uy", "p_rgh", "epsilon", "k"}

    phase = log.phases["alpha.water"]
    assert phase.fraction_first == pytest.approx(0.241)
    assert phase.fraction_last == pytest.approx(0.2404)
    assert phase.fraction_drift == pytest.approx(-6e-4, rel=1e-3)
    assert phase.minimum == pytest.approx(-1.21e-17)
    assert phase.maximum == pytest.approx(1.000002)


def test_multi_region_keeps_the_same_field_apart() -> None:
    log = FoamLog.load(LOGS / "log.chtMultiRegionFoam.regions")

    assert list(log.regions) == ["fluid", "metal"]
    assert set(log.fields) == {"fluid/h", "fluid/p_rgh", "metal/h"}
    # h in the fluid and h in the solid are different series
    assert log.residuals["fluid/h"].last_final == pytest.approx(3.88e-09)
    assert log.residuals["metal/h"].last_final == pytest.approx(1.02e-09)
    assert log.temperature_range == {
        "fluid": (302.8, 355.9),
        "metal": (303.1, 358.4),
    }
    # the per-region Courant line is printed INSIDE its step, not before it
    assert [s.courant_max for s in log.steps] == pytest.approx(
        [0.188, 0.311, 0.482, 0.611]
    )
    assert log.residual_history("metal/h")[-1].final == pytest.approx(1.02e-09)


def test_missing_boundary_condition_names_the_patch() -> None:
    log = FoamLog.load(LOGS / "log.simpleFoam.missing_bc")

    assert log.status is Status.FAILED
    assert log.reason == "missing_patch_field"
    assert log.n_steps == 0
    assert log.diagnostics[0].detail == {"patch": "outlet"}
    assert "outlet" in log.fix["note"]


def test_org_module_runner() -> None:
    log = FoamLog.load(LOGS / "log.foamRun.incompressibleFluid")

    assert log.solver == "incompressibleFluid"
    assert log.exec_command == "foamRun -solver incompressibleFluid"
    # the name is a module loaded by the runner, not the executable
    assert log.solver_source is SolverSource.MODULE
    assert log.status is Status.CONVERGED
    assert log.converged_in == 183


def test_compressible_density_bounds_are_ordered() -> None:
    log = FoamLog.load(LOGS / "log.rhoPimpleFoam.compressible")

    assert log.rho_range is not None
    low, high = log.rho_range
    # the log prints max FIRST
    assert low == pytest.approx(1.0901)
    assert high == pytest.approx(1.3251)
    assert low is not None
    assert high is not None
    assert low < high
    assert log.temperature_range == {"global": (288.15, 324.8)}
    assert log.mesh_update_time is None


def test_many_species_do_not_drown_the_summary() -> None:
    log = FoamLog.load(LOGS / "log.reactingFoam.species")

    assert len(log.fields) == 19
    assert list(log.diverging) == ["CH4"]
    # the run finished, and the answer is still worthless
    assert log.status is Status.COMPLETED
    assert log.reason == "end_reached_with_divergence"
    assert not log.healthy
    # rho under the diagonal solver was never really solved
    assert log.residuals["rho"].trend is Trend.NOT_SOLVED

    s = log.summary()
    assert len(s["residuals"]) == FoamLog.MAX_SUMMARY_RESIDUALS
    assert s["residuals_total"] == 19
    assert "CH4" in s["residuals"]
    assert s["diverging"] == ["CH4"]


def test_dynamic_mesh_reports_motion_cost() -> None:
    log = FoamLog.load(LOGS / "log.overPimpleDyMFoam.dym")
    assert log.mesh_update_time == pytest.approx(0.37)
    assert log.n_steps == 3


@pytest.mark.parametrize("name", [p.name for p in sorted(LOGS.iterdir())])
def test_summary_size_is_flat(name: str) -> None:
    """The whole point: the payload must not grow with the run."""
    s = FoamLog.load(LOGS / name).summary()
    assert "steps" not in s
    assert len(json.dumps(s)) <= 2048


def test_an_empty_log_still_answers() -> None:
    log = FoamLog.loads("")
    assert log.status is Status.INCOMPLETE
    assert log.n_steps == 0
    assert log.kind is Kind.UNKNOWN
    assert log.summary()["fields"] == []


def test_diagnostics_table_is_well_formed() -> None:
    ids = [d.id for d in DIAGNOSTICS]
    assert len(ids) == len(set(ids))
    # most of the table is confirmed against real logs; the rest is flagged so
    # a caller can weigh a hit accordingly
    from foamlib.logs import Source

    confirmed = [d for d in DIAGNOSTICS if d.source is Source.CORPUS]
    assert len(confirmed) >= 40, len(confirmed)
    # DiagnosticSpec.__post_init__ enforces group counts; assert the table size
    # so an accidental deletion is noticed
    assert len(DIAGNOSTICS) == 56


@pytest.mark.parametrize("name", [p.name for p in sorted(LOGS.iterdir())])
def test_no_placeholder_reaches_the_caller(name: str) -> None:
    for d in FoamLog.load(LOGS / name).diagnostics:
        assert "{" not in d.cause
        assert "{" not in d.fix_note
