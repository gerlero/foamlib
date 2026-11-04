"""Single-pass scanner over an OpenFOAM log.

Every regex here matches a wording seen in a real log. The ones that look
redundant are not: OpenFOAM reports the same condition differently depending on
fork, version and which part of the code printed it.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Final

from ._diagnostics import BENIGN_WARNINGS, FALLTHROUGH, DiagnosticSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

_FLOAT = r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"

# --- header ---------------------------------------------------------------
_BUILD = re.compile(r"^Build\s*:\s*(.+?)\s*$")
_EXEC = re.compile(r"^Exec\s*:\s*(.+?)\s*$")
_CASE = re.compile(r"^Case\s*:\s*(.+?)\s*$")
_NPROCS = re.compile(r"^nProcs\s*:\s*(\d+)")
_VERSION = re.compile(r"Version:\s*(\S+)")
#: older builds print "Web:", newer ones "Website:"
_WEBSITE = re.compile(r"Web(?:site)?:\s*(\S+)")
#: some wrappers print only: Using: OpenFOAM-2412 (2412) - visit www.openfoam.com
_USING = re.compile(r"^Using:\s*OpenFOAM-(\S+)")
_VERSION_IN_BUILD = re.compile(r"version=(\S+)")
_SOLVER_OPT = re.compile(r"-solver\s+(\S+)")
_SELECTING_SOLVER = re.compile(r"^Selecting solver (\S+)")
#: log.<solver> in the filename, used only when the header was stripped
_LOGNAME = re.compile(r"log\.([A-Za-z][A-Za-z0-9_]*)")

# --- time loop ------------------------------------------------------------
#: openfoam.org v11+ appends a unit: "Time = 0.001s", or "Time = 180CAD" for
#: engine cases. Older versions and the .com fork print "Time = 0.001". All must
#: match.
_TIME = re.compile(rf"^Time\s*=\s*({_FLOAT})\s*(?:s|CAD)?\s*$")
_COURANT = re.compile(rf"^Courant Number mean:\s*({_FLOAT})\s+max:\s*({_FLOAT})")
_REGION_COURANT = re.compile(
    rf"^Region: (\S+) Courant Number mean:\s*({_FLOAT})\s+max:\s*({_FLOAT})"
)
_ICOURANT = re.compile(
    rf"^Interface Courant Number mean:\s*({_FLOAT})\s+max:\s*({_FLOAT})"
)
_DELTAT = re.compile(rf"^deltaT\s*=\s*({_FLOAT})")
_SOLVE = re.compile(
    rf"Solving for ([^,]+),\s*Initial residual\s*=\s*({_FLOAT}),"
    rf"\s*Final residual\s*=\s*({_FLOAT}),\s*No Iterations\s*(\d+)"
)
_REGION = re.compile(r"^Solving for (?:fluid|solid) region (\S+)")
_PHASE = re.compile(
    rf"Phase-\d+ volume fraction = ({_FLOAT})\s+Min\((\S+)\) = ({_FLOAT})"
    rf"\s+Max\([^)]+\) = ({_FLOAT})"
)
_MINMAX_T = re.compile(rf"^Min/max T:\s*({_FLOAT})\s+({_FLOAT})")
#: ESI reactingFoam writes it differently: "min/max(T) = 300, 1200.0305"
_MINMAX_T_ALT = re.compile(rf"^min/max\((\w+)\)\s*=\s*({_FLOAT})\s*,\s*({_FLOAT})")
#: compressible solvers print max first, then min
_RHO_RANGE = re.compile(rf"^rho max/min\s*:\s*({_FLOAT})\s+({_FLOAT})")
#: dynamic-mesh and overset solvers; how much of the run is mesh motion
_MESH_UPDATE = re.compile(rf"Execution time for mesh\.update\(\)\s*=\s*({_FLOAT})\s*s")
_CONTINUITY = re.compile(rf"continuity errors\s*:.*cumulative\s*=\s*({_FLOAT})")
_BOUNDING = re.compile(r"^bounding\s+(\S+),")
_EXEC_TIME = re.compile(rf"^ExecutionTime\s*=\s*({_FLOAT})")
_CLOCK_TIME = re.compile(rf"ClockTime\s*=\s*({_FLOAT})")
_CONVERGED = re.compile(r"solution converged in (\d+) iterations")
_FATAL_IO = re.compile(r"FOAM FATAL IO ERROR")
_FATAL = re.compile(r"FOAM FATAL ERROR")
#: Whether the run is steady is a property of the case (``ddtSchemes``), not of
#: the log, so these only collect evidence. The PIMPLE loop is used for steady
#: runs too ("PIMPLE: Operating solver in steady-state mode"), and Courant is
#: not printed by every transient solver, so neither "PIMPLE" nor the absence
#: of Courant decides anything on its own. The prefix matters: potentialFoam
#: prints "potentialFlow: Operating solver in PISO mode".
_STEADY_MODE = re.compile(
    r"^(?:PIMPLE|SIMPLE): Operating solver in (?:steady-state|SIMPLE) mode"
)
_TRANSIENT_MODE = re.compile(
    r"^(?:PIMPLE|PISO): Operating solver in (?:transient|PISO) mode|^PISO:"
)
#: weaker: see SIMPLE_BUT_TRANSIENT
_SIMPLE = re.compile(r"^SIMPLE(?::| solution converged)")

#: Transient solvers that drive their non-orthogonal loop with simpleControl
#: and so print "SIMPLE:" lines whatever ``ddtSchemes`` says.
SIMPLE_BUT_TRANSIENT: Final[frozenset[str]] = frozenset(
    {"laplacianFoam", "scalarTransportFoam"}
)

#: OpenFOAM utilities that produce a log but never a time loop. Keyed off the
#: Exec name, which parses reliably, rather than off guessed output formats.
UTILITIES: Final[frozenset[str]] = frozenset(
    {
        "blockMesh",
        "checkMesh",
        "snappyHexMesh",
        "decomposePar",
        "reconstructPar",
        "reconstructParMesh",
        "setFields",
        "topoSet",
        "setSet",
        "createPatch",
        "extrudeMesh",
        "surfaceFeatureExtract",
        "surfaceFeatures",
        "transformPoints",
        "refineMesh",
        "renumberMesh",
        "splitMeshRegions",
        "mergeMeshes",
        "mapFields",
        "mapFieldsPar",
        "foamToVTK",
        "postProcess",
        "sample",
        "checkFieldsAndMesh",
        "polyDualMesh",
        "moveDynamicMesh",
        "zipUpMesh",
    }
)


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fork_of(line: str) -> str:
    """Which OpenFOAM. Real logs write the domain in several cases and
    spellings, and name the field ``Web:`` or ``Website:`` depending on age."""
    blob = line.lower()
    if "rapidcfd" in blob:
        return "rapidcfd"
    if "extend-project" in blob or "openfoam extend project" in blob:
        return "extend"
    if "openfoam.com" in blob or "openfoam=" in blob:
        return "com"
    if "openfoam.org" in blob:
        return "org"
    return "unknown"


def solver_from_filename(path: str | os.PathLike[str]) -> str | None:
    m = _LOGNAME.search(os.path.basename(os.fspath(path)))
    return m.group(1) if m else None


def _record(
    found: dict[str, Any], spec: DiagnosticSpec, line: str, lineno: int
) -> None:
    hit = found.get(spec.id)
    if hit is not None:
        hit["count"] += 1
        return
    m = spec.pattern.search(line)
    detail: dict[str, str] = {}
    if m is not None and spec.groups:
        # A regex may be an alternation where only one branch captures, so a
        # group can legitimately be None; those placeholders stay unfilled.
        detail = {n: v for n, v in zip(spec.groups, m.groups()) if v is not None}

    def fill(template: str) -> str:
        try:
            return template.format(**detail)
        except (KeyError, IndexError):
            return template

    found[spec.id] = {
        "id": spec.id,
        "severity": spec.severity,
        "cause": fill(spec.cause),
        "fix_note": fill(spec.fix_note),
        "detail": detail,
        "first_line": lineno,
        "count": 1,
    }


def _new_step() -> dict[str, Any]:
    return {
        "time": None,
        "delta_t": None,
        "courant_max": None,
        "interface_courant_max": None,
        "exec_time": None,
        "clock_time": None,
        "n_solves": 0,
    }


def scan(lines: Iterator[str]) -> dict[str, Any]:
    """One pass over the lines. Returns raw state; interpretation happens in
    :mod:`foamlib.logs._log`."""
    st: dict[str, Any] = {
        "solver": None,
        "solver_modules": [],
        "exec": None,
        "build": None,
        "version": None,
        "website": None,
        "fork": "unknown",
        "case": None,
        "n_procs": 1,
        "steps": [],
        "cur": None,
        "region": None,
        "regions": [],
        "courant_max": None,
        "interface_courant_max": None,
        "wallclock": None,
        "fields": [],
        "field_seen": set(),
        "bounding": {},
        "continuity_cumulative_last": None,
        "converged_in": None,
        "end": False,
        "fatal_io": False,
        "fatal": False,
        "last_finals": {},
        "solves_this_step": {},
        "phases": {},
        "temperature_range": {},
        "rho_range": None,
        "mesh_update": None,
        "diagnostics": {},
        "pending_courant": None,
        "pending_icourant": None,
        "pending_delta_t": None,
        "first_initial": {},
        "last_initial": {},
        # per step: (time, [(key, bare name, initial, final, iterations), ...])
        "solves": [],
        "solves_in_step": [],
        "steady_evidence": False,
        "simple_lines": False,
        "transient_evidence": False,
    }

    def close_step() -> None:
        if st["cur"] is None:
            return
        st["steps"].append(st["cur"])
        for key, val in st["solves_this_step"].items():
            st["last_finals"].setdefault(key, []).append(val)
        st["solves_this_step"] = {}
        if st["solves_in_step"]:
            st["solves"].append((st["cur"]["time"], tuple(st["solves_in_step"])))
        st["solves_in_step"] = []

    def open_step(value: float | None) -> None:
        close_step()
        step = _new_step()
        step["time"] = value
        step["courant_max"] = st["pending_courant"]
        step["interface_courant_max"] = st["pending_icourant"]
        step["delta_t"] = st["pending_delta_t"]
        st["cur"] = step
        st["pending_courant"] = None
        st["pending_icourant"] = None
        st["pending_delta_t"] = None
        st["region"] = None

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue

        if m := _TIME.match(line):
            open_step(_num(m.group(1)))
            continue

        if m := _SOLVE.search(line):
            if st["cur"] is None:
                st["cur"] = _new_step()
            key = f"{st['region']}/{m.group(1)}" if st["region"] else m.group(1)
            if key not in st["field_seen"]:
                st["field_seen"].add(key)
                st["fields"].append(key)
            st["cur"]["n_solves"] += 1
            initial, final = _num(m.group(2)), _num(m.group(3))
            st["solves_this_step"][key] = final
            st["first_initial"].setdefault(key, initial)
            st["last_initial"][key] = initial
            st["solves_in_step"].append(
                (key, m.group(1), initial, final, int(m.group(4)))
            )
            continue

        if m := _REGION.match(line):
            st["region"] = m.group(1)
            if st["region"] not in st["regions"]:
                st["regions"].append(st["region"])
            continue

        if m := _COURANT.match(line):
            # Courant and deltaT print BEFORE the "Time =" line they belong to,
            # so they are held until that step opens.
            val = _num(m.group(2))
            st["transient_evidence"] = True
            st["pending_courant"] = val
            if val is not None and (
                st["courant_max"] is None or val > st["courant_max"]
            ):
                st["courant_max"] = val
            continue

        if m := _REGION_COURANT.match(line):
            # Unlike the plain Courant line, the per-region one is printed
            # INSIDE the step it describes, so it lands on the open step.
            val = _num(m.group(3))
            st["transient_evidence"] = True
            if st["cur"] is not None:
                st["cur"]["courant_max"] = val
            else:
                st["pending_courant"] = val
            if val is not None and (
                st["courant_max"] is None or val > st["courant_max"]
            ):
                st["courant_max"] = val
            continue

        if m := _ICOURANT.match(line):
            val = _num(m.group(2))
            st["pending_icourant"] = val
            cur = st["interface_courant_max"]
            if val is not None and (cur is None or val > cur):
                st["interface_courant_max"] = val
            continue

        if m := _DELTAT.match(line):
            st["pending_delta_t"] = _num(m.group(1))
            st["transient_evidence"] = True
            continue

        if m := _PHASE.search(line):
            frac, name = _num(m.group(1)), m.group(2)
            lo, hi = _num(m.group(3)), _num(m.group(4))
            ph = st["phases"].setdefault(
                name,
                {"fraction_first": frac, "fraction_last": frac, "min": lo, "max": hi},
            )
            ph["fraction_last"] = frac
            if lo is not None and (ph["min"] is None or lo < ph["min"]):
                ph["min"] = lo
            if hi is not None and (ph["max"] is None or hi > ph["max"]):
                ph["max"] = hi
            continue

        if m := _MINMAX_T.match(line):
            region = st["region"] or "global"
            st["temperature_range"][region] = (_num(m.group(1)), _num(m.group(2)))
            continue

        if m := _MINMAX_T_ALT.match(line):
            if m.group(1) == "T":
                region = st["region"] or "global"
                st["temperature_range"][region] = (_num(m.group(2)), _num(m.group(3)))
            continue

        if m := _RHO_RANGE.match(line):
            hi, lo = _num(m.group(1)), _num(m.group(2))
            st["rho_range"] = (lo, hi)
            continue

        if m := _MESH_UPDATE.search(line):
            val = _num(m.group(1))
            if val is not None:
                st["mesh_update"] = (st["mesh_update"] or 0.0) + val
            continue

        if m := _CONTINUITY.search(line):
            st["continuity_cumulative_last"] = _num(m.group(1))
            continue

        if m := _BOUNDING.match(line):
            name = m.group(1)
            st["bounding"][name] = st["bounding"].get(name, 0) + 1
            continue

        if m := _EXEC_TIME.match(line):
            st["wallclock"] = _num(m.group(1))
            if st["cur"] is not None:
                st["cur"]["exec_time"] = st["wallclock"]
                if m2 := _CLOCK_TIME.search(line):
                    st["cur"]["clock_time"] = _num(m2.group(1))
            continue

        if _STEADY_MODE.match(line):
            st["steady_evidence"] = True
        elif _TRANSIENT_MODE.match(line):
            st["transient_evidence"] = True
        elif _SIMPLE.match(line):
            st["simple_lines"] = True

        if (m := _CONVERGED.search(line)) and st["converged_in"] is None:
            st["converged_in"] = int(m.group(1))
            continue

        if line == "End":
            st["end"] = True
            continue

        # --- header, then anything unrecognised goes past the failure table
        if st["build"] is None and (m := _BUILD.match(line)):
            st["build"] = m.group(1)
        if st["version"] is None and (m := _VERSION.search(line)):
            st["version"] = m.group(1)
        if st["website"] is None and (m := _WEBSITE.search(line)):
            st["website"] = m.group(1)
        if st["fork"] == "unknown":
            # Classify per line and keep the first verdict: a fixed-size banner
            # window gets eaten by rank preambles and debug noise before the
            # domain line appears.
            st["fork"] = fork_of(line)
        if st["version"] is None and (m := _USING.match(line)):
            st["version"] = m.group(1)
        if st["exec"] is None and (m := _EXEC.match(line)):
            st["exec"] = m.group(1)
        if st["case"] is None and (m := _CASE.match(line)):
            st["case"] = m.group(1)
        if m := _NPROCS.match(line):
            st["n_procs"] = int(m.group(1))
        if m := _SELECTING_SOLVER.match(line):
            st["solver"] = m.group(1)
            if m.group(1) not in st["solver_modules"]:
                st["solver_modules"].append(m.group(1))

        if _FATAL_IO.search(line):
            st["fatal_io"] = True
            st["fatal"] = True
        elif _FATAL.search(line):
            st["fatal"] = True

        for spec in FALLTHROUGH:
            if not spec.pattern.search(line):
                continue
            if spec.id == "foam_warning" and any(
                b.search(line) for b in BENIGN_WARNINGS
            ):
                # routine chatter, not a finding
                continue
            _record(st["diagnostics"], spec, line, lineno)

    close_step()

    if st["version"] is None and st["build"]:
        m = _VERSION_IN_BUILD.search(st["build"])
        st["version"] = m.group(1) if m else st["build"].split("-")[0]

    return st


def resolve_solver(st: dict[str, Any]) -> str:
    """Name the run. The .org runners make this three cases, not one.

    Settled against real logs, because every shortcut here was wrong:

    * ``foamMultiRun`` loads one module per region, so the run is the runner
      itself; taking the last ``Selecting solver`` line would name the run
      after whichever region happened to come last.
    * ``foamRun -solver X`` names the module on the command line, but the
      argument is absent more often than present -- the module then comes from
      ``controlDict`` and only the ``Selecting solver`` line reveals it.
    * one ``foamRun`` can select several modules (a ``functions`` run also
      instantiates the physics), so the first one selected is the one it was
      asked to run.
    """
    exec_line: str = st["exec"] or ""
    parts = exec_line.split()
    # clusters and containers put the full path in Exec:
    #   Exec : /home1/apps/.../platforms/linux64.../bin/snappyHexMesh
    first = os.path.basename(parts[0]) if parts else ""

    if first == "foamMultiRun":
        return first
    if m := _SOLVER_OPT.search(exec_line):
        return m.group(1)
    if st["solver_modules"]:
        return str(st["solver_modules"][0])
    return first
