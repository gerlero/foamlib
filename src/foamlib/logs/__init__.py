"""
The logs module turns an OpenFOAM log into a structured verdict.

It reads solver and utility logs of any size, names the failure when the run
failed, and keeps the summary the same size whatever the run length.

"""

from ._diagnostics import DIAGNOSTICS, DiagnosticSpec, Severity, Source
from ._log import (
    SOLVED_TOL,
    Diagnostic,
    FoamLog,
    Kind,
    PhaseInfo,
    Residual,
    ResidualHistory,
    SolverSource,
    Status,
    Step,
    Trend,
)

__all__ = [
    "DIAGNOSTICS",
    "SOLVED_TOL",
    "Diagnostic",
    "DiagnosticSpec",
    "FoamLog",
    "Kind",
    "PhaseInfo",
    "Residual",
    "ResidualHistory",
    "Severity",
    "SolverSource",
    "Source",
    "Status",
    "Step",
    "Trend",
]
