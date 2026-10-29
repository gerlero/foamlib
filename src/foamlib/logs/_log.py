"""Structured reading of OpenFOAM solver logs.

An OpenFOAM log is sized for a disk, not for a reader: a ten-thousand-step run
is tens of megabytes. :class:`FoamLog` turns one into a verdict -- did it
converge, why not, and what to change -- and :meth:`FoamLog.summary` keeps that
verdict the same size whatever the run length, so it can be handed to something
with a budget.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from . import _scan
from ._diagnostics import BY_ID, PRE_EXEC_REASONS, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    if sys.version_info >= (3, 14):
        from io import Reader
    else:
        from typing_extensions import Reader

__all__ = [
    "Diagnostic",
    "FoamLog",
    "Kind",
    "PhaseInfo",
    "Residual",
    "ResidualHistory",
    "Status",
    "Step",
    "Trend",
]

#: Default floor below which a final residual counts as solved, whatever the
#: ratio says. Real logs routinely start a field at exactly 0, which made a pure
#: ratio test call a converged 1e-9 "diverging". The log never states the
#: case's own ``tolerance`` (it lives in ``system/fvSolution``), so pass
#: ``tolerance=`` to the constructors when it is known.
SOLVED_TOL: Final = 1e-6


class Kind(StrEnum):
    """What the log is a record of."""

    SOLVER = "solver"
    UTILITY = "utility"
    LAUNCHER_FAILURE = "launcher_failure"
    UNKNOWN = "unknown"


class Status(StrEnum):
    """How the run ended.

    This answers "did the executable finish", not "is the answer any good":
    :attr:`FoamLog.reason` and :attr:`FoamLog.diverging` answer that.
    """

    CONVERGED = "converged"
    COMPLETED = "completed"
    DIVERGED = "diverged"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class Trend(StrEnum):
    """Where a field's residuals were heading, worst first."""

    DIVERGING = "diverging"
    STALLED = "stalled"
    UNKNOWN = "unknown"
    CONVERGING = "converging"
    NOT_SOLVED = "not_solved"


class SolverSource(StrEnum):
    """Where the solver name came from, so it can be trusted accordingly."""

    #: the solver is its own executable, as for the classic solvers
    EXEC = "exec"
    #: a module loaded by the .org ``foamRun``/``foamMultiRun`` runners
    MODULE = "module"
    #: inferred from the filename because the header was stripped or absent
    FILENAME = "filename"
    #: no evidence at all
    UNKNOWN = ""


#: How much an agent should care, lowest first.
_TREND_RANK: Final[Mapping[Trend, int]] = {
    Trend.DIVERGING: 0,
    Trend.STALLED: 1,
    Trend.UNKNOWN: 2,
    Trend.CONVERGING: 3,
    Trend.NOT_SOLVED: 4,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Diagnostic:
    """A named failure mode, with the specifics pulled out of the log."""

    id: str
    severity: Severity
    cause: str
    fix_note: str
    detail: Mapping[str, str] = field(default_factory=dict)
    first_line: int = 0
    count: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class Residual:
    """One field's residuals within one time step."""

    time: float | None
    initial: float | None
    final: float | None
    iterations: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidualHistory:
    """How one field behaved over the whole run."""

    first_initial: float | None
    last_initial: float | None
    first_final: float | None
    last_final: float | None
    trend: Trend


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """One time step."""

    time: float | None
    delta_t: float | None
    courant_max: float | None
    interface_courant_max: float | None
    exec_time: float | None
    clock_time: float | None
    n_solves: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseInfo:
    """Phase-fraction bookkeeping for a VoF run: mass conservation, in effect."""

    fraction_first: float | None
    fraction_last: float | None
    minimum: float | None
    maximum: float | None

    @property
    def fraction_drift(self) -> float | None:
        if self.fraction_first is None or self.fraction_last is None:
            return None
        return self.fraction_last - self.fraction_first


def _trend(
    first_final: float | None, last_final: float | None, n: int, tol: float
) -> Trend:
    if n < 2 or first_final is None or last_final is None:
        return Trend.UNKNOWN
    if first_final == 0.0 and last_final == 0.0:
        # e.g. rho under the `diagonal` solver: zero residual, zero iterations.
        # Not stalled, just never really solved -- and it must not crowd real
        # fields out of a capped summary.
        return Trend.NOT_SOLVED
    # absolute verdicts first; they need no reference value
    if last_final <= tol:
        return Trend.CONVERGING
    if last_final > 1.0:
        return Trend.DIVERGING
    # relative verdicts need a usable starting point
    if first_final > 0.0:
        if last_final > first_final * 100:
            return Trend.DIVERGING
        if last_final < first_final / 10:
            return Trend.CONVERGING
    return Trend.STALLED


@dataclass(frozen=True, slots=True, kw_only=True)
class FoamLog:
    """A parsed OpenFOAM log.

    Build one with :meth:`read` (a path), :meth:`load` (an open file) or
    :meth:`loads` (a string)::

        log = FoamLog.read("log.simpleFoam")
        if log.status is not Status.CONVERGED:
            print(log.reason, log.fix)
    """

    #: Default caps for :meth:`summary`, chosen so the payload stays ~1-2 kB.
    MAX_SUMMARY_RESIDUALS: ClassVar[int] = 8
    MAX_SUMMARY_DIAGNOSTICS: ClassVar[int] = 3

    #: the filename the log came from, if known; see :meth:`loads`
    name: str | None
    kind: Kind
    status: Status
    reason: str
    solver: str
    solver_source: SolverSource
    solver_modules: Sequence[str]
    exec_command: str
    fork: str
    version: str
    build: str
    case: str
    n_procs: int
    #: ``None`` when the log carries no evidence either way; see :meth:`loads`
    steady: bool | None
    n_steps: int
    last_time: float | None
    courant_max: float | None
    interface_courant_max: float | None
    wallclock: float | None
    mesh_update_time: float | None
    fields: Sequence[str]
    regions: Sequence[str]
    residuals: Mapping[str, ResidualHistory]
    diverging: Sequence[str]
    phases: Mapping[str, PhaseInfo]
    temperature_range: Mapping[str, tuple[float | None, float | None]]
    rho_range: tuple[float | None, float | None] | None
    bounding: Mapping[str, int]
    continuity_cumulative_last: float | None
    converged_in: int | None
    diagnostics: Sequence[Diagnostic]
    fix: Mapping[str, Any]
    steps: Sequence[Step]
    _solves: Sequence[
        tuple[float | None, Sequence[tuple[str, str, float | None, float | None, int]]]
    ] = field(default=(), repr=False, compare=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def read(
        cls,
        path: os.PathLike[str] | str,
        /,
        *,
        tolerance: float | Mapping[str, float] | None = None,
    ) -> FoamLog:
        """Parse the log file at ``path``, streaming it line by line."""
        with Path(path).open("r", errors="replace") as fp:
            return cls._build(_scan.scan(fp), name=os.fspath(path), tolerance=tolerance)

    @classmethod
    def load(
        cls,
        fp: Reader[bytes] | Reader[str],
        /,
        *,
        name: str | None = None,
        tolerance: float | Mapping[str, float] | None = None,
    ) -> FoamLog:
        """Parse a log from an open file, text or binary.

        ``name`` defaults to ``fp.name`` when the file has one; see :meth:`loads`.
        """
        if name is None and isinstance(fp_name := getattr(fp, "name", None), str):
            name = fp_name
        return cls.loads(fp.read(), name=name, tolerance=tolerance)

    @classmethod
    def loads(
        cls,
        s: bytes | bytearray | str,
        /,
        *,
        name: str | None = None,
        tolerance: float | Mapping[str, float] | None = None,
    ) -> FoamLog:
        """Parse a log held in memory.

        :param name: The filename the log was saved under, such as
            ``"log.simpleFoam"``. It is never opened. It is only consulted when
            the header naming the solver was stripped or blanked, in which case
            the solver is taken from a ``log.<solver>`` filename and
            :attr:`solver_source` says so. ``None`` skips that fallback.
        :param tolerance: The solver tolerance from ``system/fvSolution``, either
            one value or one per field name. A final residual at or below it
            counts as solved. Fields not listed use :data:`SOLVED_TOL`.

        :attr:`steady` is ``True`` or ``False`` only when the log says so: a
        SIMPLE or steady-state-mode line for steady (not from laplacianFoam or
        scalarTransportFoam, which print SIMPLE lines either way); Courant,
        ``deltaT``, PISO or transient-mode lines for transient. With no
        evidence, or evidence both ways, it is ``None``: the authoritative
        answer is ``ddtSchemes`` in the case, not the log.
        """
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode(errors="replace")
        return cls._build(
            _scan.scan(iter(s.splitlines())), name=name, tolerance=tolerance
        )

    @classmethod
    def _build(
        cls,
        st: dict[str, Any],
        *,
        name: str | None,
        tolerance: float | Mapping[str, float] | None,
    ) -> FoamLog:
        steps = [Step(**s) for s in st["steps"]]

        def tol_for(key: str) -> float:
            if tolerance is None:
                return SOLVED_TOL
            if isinstance(tolerance, (int, float)):
                return float(tolerance)
            # a region's field ("fluid/h") falls back to its bare name ("h")
            bare = key.rsplit("/", 1)[-1]
            return tolerance.get(key, tolerance.get(bare, SOLVED_TOL))

        residuals: dict[str, ResidualHistory] = {}
        for key in st["fields"]:
            series = st["last_finals"].get(key, [])
            if not series:
                continue
            residuals[key] = ResidualHistory(
                first_initial=st["first_initial"].get(key),
                last_initial=st["last_initial"].get(key),
                first_final=series[0],
                last_final=series[-1],
                trend=_trend(series[0], series[-1], len(series), tol_for(key)),
            )

        diags = dict(st["diagnostics"])
        if st["bounding"]:
            worst = max(st["bounding"], key=lambda k: st["bounding"][k])
            spec = BY_ID["bounding"]
            diags["bounding"] = {
                "id": "bounding",
                "severity": Severity.WARNING,
                "cause": spec.cause,
                "fix_note": spec.fix_note.format(field=worst),
                "detail": dict(st["bounding"]),
                "first_line": 0,
                "count": sum(st["bounding"].values()),
            }
        ordered = sorted(
            diags.values(),
            key=lambda d: (
                0 if d["severity"] is Severity.FATAL else 1,
                d["first_line"],
            ),
        )
        diagnostics = [Diagnostic(**d) for d in ordered]
        fatal = [d for d in diagnostics if d.severity is Severity.FATAL]
        diverging = sorted(
            k for k, v in residuals.items() if v.trend is Trend.DIVERGING
        )

        status, reason = _verdict(st, fatal, diverging)
        fix = _fix_for(st, fatal, status, diverging, diagnostics)

        solver = _scan.resolve_solver(st)
        source = SolverSource.EXEC if solver else SolverSource.UNKNOWN
        if solver and solver in st["solver_modules"]:
            source = SolverSource.MODULE
        if not solver:
            # Real logs exist with the header fields blanked out, or with no
            # header at all; the filename is then the only evidence, and the
            # caller should be told it was inferred.
            inferred = _scan.solver_from_filename(name) if name else None
            if inferred:
                solver, source = inferred, SolverSource.FILENAME

        steady_ev = st["steady_evidence"] or (
            st["simple_lines"] and solver not in _scan.SIMPLE_BUT_TRANSIENT
        )
        steady: bool | None = None
        if steady_ev != st["transient_evidence"]:
            steady = steady_ev
        if reason in PRE_EXEC_REASONS:
            kind = Kind.LAUNCHER_FAILURE
        elif solver in _scan.UTILITIES:
            kind = Kind.UTILITY
        elif solver or (steps and residuals):
            kind = Kind.SOLVER
        else:
            kind = Kind.UNKNOWN

        return cls(
            name=name,
            kind=kind,
            status=status,
            reason=reason,
            solver=solver,
            solver_source=source,
            solver_modules=tuple(st["solver_modules"]),
            exec_command=st["exec"] or "",
            fork=st["fork"],
            version=st["version"] or "",
            build=st["build"] or "",
            case=st["case"] or "",
            n_procs=st["n_procs"],
            steady=steady,
            n_steps=len(steps),
            last_time=steps[-1].time if steps else None,
            courant_max=st["courant_max"],
            interface_courant_max=st["interface_courant_max"],
            wallclock=st["wallclock"],
            mesh_update_time=st["mesh_update"],
            fields=tuple(st["fields"]),
            regions=tuple(st["regions"]),
            residuals=residuals,
            diverging=tuple(diverging),
            phases={
                k: PhaseInfo(
                    fraction_first=v["fraction_first"],
                    fraction_last=v["fraction_last"],
                    minimum=v["min"],
                    maximum=v["max"],
                )
                for k, v in st["phases"].items()
            },
            temperature_range=dict(st["temperature_range"]),
            rho_range=st["rho_range"],
            bounding=dict(st["bounding"]),
            continuity_cumulative_last=st["continuity_cumulative_last"],
            converged_in=st["converged_in"],
            diagnostics=tuple(diagnostics),
            fix=fix,
            steps=tuple(steps),
            _solves=tuple(st["solves"]),
        )

    # -- drill-down --------------------------------------------------------

    def residual_history(
        self, field: str, *, last: int | None = None
    ) -> list[Residual]:
        """One row per time step for ``field``, optionally only the final
        ``last`` of them. ``field`` matches either the composite key
        (``"fluid/h"``) or the bare name (``"h"``). Within a step, ``initial``
        is from the first solve of the field and ``final`` from the last."""
        rows: list[Residual] = []
        for time, solves in self._solves:
            hits = [x for x in solves if field in {x[0], x[1]}]
            if time is None or not hits:
                continue
            rows.append(
                Residual(
                    time=time,
                    initial=hits[0][2],
                    final=hits[-1][3],
                    iterations=hits[-1][4],
                )
            )
        return rows[-last:] if last is not None else rows

    # -- the bounded payload ----------------------------------------------

    def summary(
        self,
        *,
        max_residuals: int | None = None,
        max_diagnostics: int | None = None,
    ) -> dict[str, Any]:
        """A JSON-ready verdict whose size does not grow with the run.

        Lists are capped by how alarming their entries are, and ``*_total``
        gives the true count. Per-step rows are never included: use
        :attr:`steps` or :meth:`residual_history` for those.
        """
        n_res = self.MAX_SUMMARY_RESIDUALS if max_residuals is None else max_residuals
        n_diag = (
            self.MAX_SUMMARY_DIAGNOSTICS if max_diagnostics is None else max_diagnostics
        )

        worst = sorted(
            self.residuals.items(),
            key=lambda kv: (_TREND_RANK[kv[1].trend], kv[0]),
        )
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "solver": self.solver,
            "solver_source": self.solver_source.value,
            "solver_modules": list(self.solver_modules),
            "exec": self.exec_command,
            "fork": self.fork,
            "version": self.version,
            "build": self.build,
            "case": self.case,
            "n_procs": self.n_procs,
            "steady": self.steady,
            "status": self.status.value,
            "reason": self.reason,
            "n_steps": self.n_steps,
            "last_time": self.last_time,
            "courant_max": self.courant_max,
            "interface_courant_max": self.interface_courant_max,
            "wallclock_s": self.wallclock,
            "mesh_update_s": self.mesh_update_time,
            "fields": list(self.fields[:n_res]),
            "regions": list(self.regions),
            "diverging": list(self.diverging),
            "bounding": dict(self.bounding),
            "continuity_cumulative_last": self.continuity_cumulative_last,
            "converged_in": self.converged_in,
            "rho_range": list(self.rho_range) if self.rho_range else None,
            "temperature_range": {
                k: list(v) for k, v in self.temperature_range.items()
            },
            "phases": {
                k: {
                    "fraction_first": v.fraction_first,
                    "fraction_last": v.fraction_last,
                    "min": v.minimum,
                    "max": v.maximum,
                }
                for k, v in self.phases.items()
            },
            "residuals": {
                k: {
                    "first_initial": v.first_initial,
                    "last_initial": v.last_initial,
                    "first_final": v.first_final,
                    "last_final": v.last_final,
                    "trend": v.trend.value,
                }
                for k, v in worst[:n_res]
            },
            "residuals_total": len(self.residuals),
            # The caller needs the machine-readable id, how often it fired and
            # any extracted specifics. The prose (`cause`) and per-diagnostic
            # advice (`fix_note`) are redundant here: the governing action is
            # already in `fix`, and the full records stay on `diagnostics`.
            "diagnostics": [
                {
                    "id": d.id,
                    "severity": d.severity.value,
                    "count": d.count,
                    **({"detail": dict(d.detail)} if d.detail else {}),
                }
                for d in self.diagnostics[:n_diag]
            ],
            "diagnostics_total": len(self.diagnostics),
            "fix": dict(self.fix),
        }
        return out

    @property
    def healthy(self) -> bool:
        """``False`` when the run failed, diverged, or finished dirty."""
        return (
            self.status not in {Status.FAILED, Status.DIVERGED} and not self.diverging
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(solver={self.solver!r}, "
            f"status={self.status.value!r}, reason={self.reason!r}, "
            f"n_steps={self.n_steps})"
        )


def _verdict(
    st: dict[str, Any], fatal: list[Diagnostic], diverging: list[str]
) -> tuple[Status, str]:
    if fatal:
        return Status.FAILED, fatal[0].id
    if st["fatal_io"]:
        return Status.FAILED, "fatal_io_error"
    if st["fatal"]:
        return Status.FAILED, "fatal_error"
    if st["converged_in"] is not None:
        return Status.CONVERGED, "criteria_reached"
    if st["end"]:
        # The run did finish, so the status is factual -- but a diverging field
        # means the answer is worthless, and a caller that reads only `status`
        # must not miss that.
        return Status.COMPLETED, (
            "end_reached_with_divergence" if diverging else "end_reached"
        )
    if diverging:
        return Status.DIVERGED, "residual_divergence"
    return Status.INCOMPLETE, "truncated_log"


def _fix_for(
    st: dict[str, Any],
    fatal: list[Diagnostic],
    status: Status,
    diverging: list[str],
    diagnostics: Sequence[Diagnostic],
) -> dict[str, Any]:
    if fatal:
        top = fatal[0]
        fix: dict[str, Any] = dict(BY_ID[top.id].fix_extra)
        fix["note"] = top.fix_note
        if top.id == "fp_exception" and (st["courant_max"] or 0) > 1:
            # A Courant blow-up is a far more actionable diagnosis than the
            # generic floating-point advice.
            return {
                "adjustTimeStep": True,
                "maxCo": 0.9,
                "note": (
                    f"Courant number reached {st['courant_max']:g} before the "
                    f"crash; limit the time step"
                ),
            }
        return fix
    if status is Status.DIVERGED:
        return {
            "relaxationFactors": "reduce",
            "note": "residuals diverging for " + ", ".join(diverging[:4]),
        }
    if diagnostics:
        return {"note": diagnostics[0].fix_note}
    return {}
