Example
=======

This example script sets up and runs a validation test case for the ``scalarTransportFoam`` solver, verifying the diffusion of a scalar field in a simplified 2D domain.

Overview
--------

- Creates a clean OpenFOAM case in the ``diffusionCheck`` subdirectory.
- Configures mesh geometry, solver settings, and initial/boundary conditions for scalar (``T``) and velocity (``U``) fields.
- Simulates a velocity-driven scalar transport where a temperature gradient is imposed across the inlet.
- Uses :class:`foamlib.FoamCase` and related utilities to manage OpenFOAM input files and execution.
- Computes an analytical solution using the complementary error function (:func:`scipy.special.erfc`) and compares it against numerical results.

Code
----
.. literalinclude :: ../examples/basic/diffusioncheck.py
    :language: python
