📜 Solver logs (:py:mod:`foamlib.logs`)
========================================

.. note:: This module is new and its API may still change.

An OpenFOAM log is written for a disk, not for a reader: a long run produces megabytes of residual lines. :class:`foamlib.logs.FoamLog` reads a log once and answers the questions usually asked of it: did the run finish, did it converge, and if not, why and what to change.

It works on logs from the openfoam.org and openfoam.com distributions, foam-extend and RapidCFD, including logs from utilities such as ``blockMesh``, ``snappyHexMesh`` or ``checkMesh``.

Reading a log
-------------

.. code-block:: python

    from foamlib.logs import FoamLog, Status

    log = FoamLog.read("log.pimpleFoam")

    if log.status is not Status.CONVERGED:
        print(log.reason)  # e.g. "fp_exception"
        print(log.fix)     # e.g. {"adjustTimeStep": True, "maxCo": 0.9, "note": ...}

:meth:`~foamlib.logs.FoamLog.read` takes a path, :meth:`~foamlib.logs.FoamLog.load` an open text or binary file, and :meth:`~foamlib.logs.FoamLog.loads` a ``str`` or ``bytes``. All three give the same result.

Summary
-------

:meth:`~foamlib.logs.FoamLog.summary` returns a JSON-ready dictionary whose size does not grow with the length of the run (a few kB at most). Lists in it are capped, keeping the most alarming entries first, which makes it suitable for passing to a script, a dashboard or an LLM agent. For a crashed run, part of it looks like:

.. code-block:: json

    {
        "kind": "solver",
        "solver": "pimpleFoam",
        "steady": false,
        "status": "failed",
        "reason": "fp_exception",
        "n_steps": 6,
        "courant_max": 41182.0,
        "diverging": ["Ux", "Uy", "epsilon", "k", "p"],
        "diagnostics": [
            {"id": "fp_exception", "severity": "fatal", "count": 2},
            {"id": "mpi_abort", "severity": "fatal", "count": 1}
        ],
        "fix": {
            "adjustTimeStep": true,
            "maxCo": 0.9,
            "note": "Courant number reached 41182 before the crash; limit the time step"
        }
    }

Per-step data
-------------

Per-step values are kept out of the summary, but remain available:

.. code-block:: python

    log.steps                               # time, deltaT, Courant, execution time per step
    log.residual_history("p", last=50)      # initial/final residuals of one field per step
    log.residual_history("fluid/h")         # multi-region runs use "region/field"

What the log cannot tell
------------------------

Some properties belong to the case rather than to the log, so they are only reported as far as the log shows them:

- :attr:`~foamlib.logs.FoamLog.steady` is ``None`` unless the log contains evidence either way (SIMPLE or steady-state-mode lines, versus Courant, ``deltaT`` or PISO lines). The authoritative answer is ``ddtSchemes`` in ``system/fvSchemes``.
- Residual trends treat a final residual at or below :data:`foamlib.logs.SOLVED_TOL` as solved. Pass ``tolerance=`` (a single value or one per field) with the solver tolerances from ``system/fvSolution`` to use the case's own values.

Failure modes
-------------

:data:`foamlib.logs.DIAGNOSTICS` is the table of named failure modes (missing files, boundary condition errors, floating-point exceptions, MPI aborts, meshing failures, ...). Each entry records whether it was confirmed against real logs (:attr:`foamlib.logs.Source.CORPUS`) or only expected from how OpenFOAM behaves and not yet seen in one (:attr:`foamlib.logs.Source.INFERRED`).

API reference
-------------

.. autoclass:: foamlib.logs.FoamLog
   :members: read, load, loads, summary, residual_history, healthy

.. autoclass:: foamlib.logs.Status
   :members:
   :undoc-members:

.. autoclass:: foamlib.logs.Kind
   :members:
   :undoc-members:

.. autoclass:: foamlib.logs.Trend
   :members:
   :undoc-members:

.. autoclass:: foamlib.logs.SolverSource
   :members:
   :undoc-members:

.. autoclass:: foamlib.logs.Diagnostic
.. autoclass:: foamlib.logs.ResidualHistory
.. autoclass:: foamlib.logs.Residual
.. autoclass:: foamlib.logs.Step
.. autoclass:: foamlib.logs.PhaseInfo
.. autoclass:: foamlib.logs.DiagnosticSpec
.. autoclass:: foamlib.logs.Severity
   :members:
   :undoc-members:

.. autoclass:: foamlib.logs.Source
   :members:
   :undoc-members:

.. autodata:: foamlib.logs.SOLVED_TOL
.. autodata:: foamlib.logs.DIAGNOSTICS
   :no-value:
