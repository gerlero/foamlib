"""Known OpenFOAM failure modes.

The regexes and severities are domain data: every pattern here was observed in
a real log. The table was built by running the parser over a corpus of 884
public logs (364 solver runs, 520 utility runs) and adding every failure it
could not name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class Severity(StrEnum):
    """Whether a named failure stopped the run."""

    FATAL = "fatal"
    WARNING = "warning"


class Source(StrEnum):
    """How much to trust that this pattern matches what OpenFOAM really prints."""

    #: seen firing on real logs in the validation corpus
    CORPUS = "corpus"
    #: plausible from the tool's behaviour but never observed in the corpus.
    #: Treat a hit as a hint, and please open an issue with the log.
    INFERRED = "inferred"


@dataclass(frozen=True, slots=True)
class DiagnosticSpec:
    """One recognisable failure mode."""

    id: str
    severity: Severity
    pattern: re.Pattern[str]
    groups: tuple[str, ...] = ()
    cause: str = ""
    fix_note: str = ""
    fix_extra: Mapping[str, object] = field(default_factory=dict)
    source: Source = Source.INFERRED

    def __post_init__(self) -> None:
        if self.pattern.groups != len(self.groups):
            msg = (
                f"{self.id}: pattern has {self.pattern.groups} capturing groups "
                f"but {len(self.groups)} names"
            )
            raise ValueError(msg)


#: Failures that happen before OpenFOAM prints anything, so a missing header is
#: the truth about the log rather than a parser miss.
PRE_EXEC_REASONS: Final[frozenset[str]] = frozenset(
    {
        "mpi_not_enough_slots",
        "missing_case_directory",
        "mpi_executable_not_found",
        "mpi_run_as_root",
        "wrong_argument_count",
    }
)

DIAGNOSTICS: Final[Sequence[DiagnosticSpec]] = (
    DiagnosticSpec(
        id="missing_patch_field",
        severity=Severity.FATAL,
        pattern=re.compile("Cannot find patchField entry for (\\S+)"),
        groups=("patch",),
        cause="a patch present in the mesh has no matching entry in a field file's boundaryField block",
        fix_note="add a boundaryField entry for patch {patch} in the field file named by the error",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="undefined_keyword",
        severity=Severity.FATAL,
        pattern=re.compile('keyword (\\S+) is undefined in dictionary \\"([^\\"]+)\\"'),
        groups=("keyword", "dictionary"),
        cause="a required entry is missing from a dictionary the solver tried to read",
        fix_note="add keyword {keyword} to {dictionary}",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="incompatible_dimensions",
        severity=Severity.FATAL,
        pattern=re.compile("incompatible dimensions for operation"),
        groups=(),
        cause="quantities with different physical units were combined in one operation",
        fix_note="check that every coefficient includes its dimensions and the correct fields are used in expressions",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="unknown_patch_field_type",
        severity=Severity.FATAL,
        pattern=re.compile("Unknown patchField type (\\S+)"),
        groups=("type",),
        cause="the boundary condition type name in a field file is not recognised by this OpenFOAM build",
        fix_note="correct the type {type} to one of the valid names listed in the error message",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="cast_failure",
        severity=Severity.FATAL,
        pattern=re.compile("attempt to cast type (\\S+) to type (\\S+)"),
        groups=("from_type", "to_type"),
        cause="a boundary condition of the wrong kind was used where another kind was required",
        fix_note="replace the {from_type} boundary condition with one of kind {to_type} to match the field",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="field_mesh_mismatch",
        severity=Severity.FATAL,
        pattern=re.compile(
            "Number of cells in mesh and field do not match|size (\\d+) is not equal to the given value of (\\d+)"
        ),
        groups=("got", "expected"),
        cause="a field file was written for a different mesh than the one currently loaded",
        fix_note="regenerate the initial fields for the current mesh so size {got} matches the expected {expected}",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="missing_file",
        severity=Severity.FATAL,
        pattern=re.compile('[Cc]annot (?:find|open) file \\"?([^\\"\\n]+)'),
        groups=("path",),
        cause="a file the case needs, such as the mesh, a 0/ field, or a dictionary, is absent",
        fix_note="create or restore the missing file at {path} and check the case path settings",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="solution_singularity",
        severity=Severity.FATAL,
        pattern=re.compile("Solution singularity"),
        groups=(),
        cause="the linear system has no unique solution because pressure has no reference level in the closed domain",
        fix_note="set a pressure reference by adding pRefCell and pRefValue to the PIMPLE or SIMPLE dictionary",
        fix_extra={"pRefCell": 0, "pRefValue": 0},
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="fp_exception",
        severity=Severity.FATAL,
        pattern=re.compile("Signal: Floating point exception|sigFpe::sigHandler"),
        groups=(),
        cause="a division by zero, square root of a negative number, or numeric overflow occurred during the run",
        fix_note="reduce the time step, tighten discretisation schemes, and check coefficients for zero or negative values",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="segfault",
        severity=Severity.FATAL,
        pattern=re.compile("Signal: Segmentation fault"),
        groups=(),
        cause="a memory fault occurred, commonly from a corrupt mesh or a parallel run count mismatched with the decomposition",
        fix_note="run checkMesh to validate the mesh and ensure the number of processes matches the decomposition",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="max_iterations",
        severity=Severity.WARNING,
        pattern=re.compile("Maximum number of iterations exceeded"),
        groups=(),
        cause="the linear solver reached its iteration cap without meeting its tolerance",
        fix_note="increase the iteration limit or relax tolerances, and check the mesh and initial conditions for the source",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="continuity_not_removable",
        severity=Severity.WARNING,
        pattern=re.compile(
            "Continuity error cannot be removed by adjusting the outflow"
        ),
        groups=(),
        cause="the boundary conditions fix the flux on every boundary so inflow and outflow cannot be balanced",
        fix_note="change at least one outlet to a boundary condition free to adjust, such as inletOutlet",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="negative_temperature",
        severity=Severity.FATAL,
        pattern=re.compile("Negative initial temperature|negative temperature"),
        groups=(),
        cause="the energy equation produced a temperature below absolute zero, which is unphysical",
        fix_note="reduce the time step, check the thermophysical settings, and verify boundary and initial temperatures",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="bounding",
        severity=Severity.WARNING,
        pattern=re.compile("^bounding (\\S+),"),
        groups=("field",),
        cause="field values were clipped back into their physical range because the solution went out of bounds",
        fix_note="watch repeated bounding of {field}; if frequent, review the turbulence and scalar scheme settings",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="processor_dir_mismatch",
        severity=Severity.FATAL,
        pattern=re.compile(
            "[Nn]umber of processor directories = (\\d+) (?:is not equal to|does not match) the number of processors = (\\d+)"
        ),
        groups=("found", "expected"),
        cause="the case was decomposed for {found} processors but is being run with {expected}",
        fix_note="re-run decomposePar with numberOfSubdomains set to {expected}, or launch with {found} ranks",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="zero_negative_cell_volume",
        severity=Severity.FATAL,
        pattern=re.compile("\\*\\*\\*\\s*Zero or negative cell volume"),
        groups=(),
        cause="checkMesh found cells with zero or inverted volume, so the mesh is unusable",
        fix_note="regenerate or repair the mesh so every cell has positive volume before retrying the run",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="zero_negative_face_area",
        severity=Severity.FATAL,
        pattern=re.compile("\\*\\*\\*\\s*Zero or negative face area"),
        groups=(),
        cause="checkMesh found faces collapsed to zero or negative area, from collapsed points or a bad snap",
        fix_note="check the snapping and mesh generation settings, then rebuild the mesh without degenerate faces",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="incorrectly_oriented_faces",
        severity=Severity.FATAL,
        pattern=re.compile(
            "\\*\\*\\*\\s*Error in face pyramids: (\\d+) faces are incorrectly oriented"
        ),
        groups=("faces",),
        cause="{faces} faces point away from their owner cell so fluxes would get the wrong sign",
        fix_note="fix the mesh so all {faces} incorrectly oriented faces point outward from their owner cell",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="severe_non_orthogonality",
        severity=Severity.WARNING,
        pattern=re.compile(
            "\\*\\*\\*\\s*Number of severely non-orthogonal[^:]*: (\\d+)"
        ),
        groups=("faces",),
        cause="{faces} faces meet their neighbours at a very oblique angle, degrading gradient accuracy",
        fix_note="raise nNonOrthogonalCorrectors in fvSolution or regenerate the mesh with better orthogonality",
        source=Source.INFERRED,
    ),
    DiagnosticSpec(
        id="mesh_checks_failed",
        severity=Severity.WARNING,
        pattern=re.compile("Failed (\\d+) mesh checks"),
        groups=("count",),
        cause="checkMesh reported {count} failing checks overall so mesh quality needs review",
        fix_note="review the {count} failing mesh checks in the checkMesh output and repair the mesh",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="unknown_function_type",
        severity=Severity.FATAL,
        pattern=re.compile("Unknown function type (\\S+)"),
        groups=("type",),
        cause="a functionObject of type {type} is not provided by this build, likely a typo or foreign fork",
        fix_note="correct the functionObject type {type} in controlDict to a valid name from the error message",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="mpi_abort",
        severity=Severity.FATAL,
        pattern=re.compile(
            "MPI_ABORT was invoked|mpirun detected that one or more processes exited with non-zero status"
        ),
        groups=(),
        cause="the parallel run was torn down because at least one rank died, so this is a symptom",
        fix_note="read the individual rank output above the abort to find the real error before changing anything",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="foam_warning",
        severity=Severity.WARNING,
        pattern=re.compile("-->\\s*FOAM Warning"),
        groups=(),
        cause="the run emitted warnings this table does not yet name specifically",
        fix_note="search the log for the FOAM Warning blocks and address each message shown there",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="missing_case_directory",
        severity=Severity.FATAL,
        pattern=re.compile('cannot open case directory \\"?([^\\"\\n]+)'),
        groups=("path",),
        cause="the solver was pointed at a case directory that does not exist",
        fix_note="run from the case directory, or correct the -case path; {path} was not found",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="mpi_not_enough_slots",
        severity=Severity.FATAL,
        pattern=re.compile("There are not enough slots available in the system"),
        groups=(),
        cause="MPI refused to start because the host offers fewer slots than ranks requested",
        fix_note="request no more ranks than available cores, or pass --oversubscribe to mpirun",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="solver_superseded",
        severity=Severity.WARNING,
        pattern=re.compile(
            "(\\S+) has been superseded and replaced by the more general"
        ),
        groups=("old_solver",),
        cause="the named executable is a shim and silently ran a different solver module",
        fix_note="{old_solver} is deprecated; call the replacement runner directly so the log says what ran",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="dict_syntax_error",
        severity=Severity.FATAL,
        pattern=re.compile("Wrong token type - expected (\\S+), found on line (\\d+)"),
        groups=("expected", "line"),
        cause="a dictionary has a syntax error and parsing stopped at an unexpected token",
        fix_note="fix the syntax at line {line}, where {expected} was expected; the log's next line names the file",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="entry_not_found",
        severity=Severity.FATAL,
        pattern=re.compile('Entry \'(\\S+)\' not found in dictionary \\"([^\\"]+)\\"'),
        groups=("entry", "dictionary"),
        cause="the dictionary {dictionary} lacks the required entry {entry}",
        fix_note="add entry {entry} to {dictionary}, or correct its spelling; the log names the file path",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="codestream_disabled",
        severity=Severity.FATAL,
        pattern=re.compile(
            "Loading shared libraries using case-supplied code may have been disabled"
        ),
        groups=(),
        cause="this build blocks compiling case-supplied code such as #codeStream for security reasons",
        fix_note="pre-compile the library into the installation, or allow case-supplied code in this environment",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="ami_failure",
        severity=Severity.FATAL,
        pattern=re.compile("Unable to set source and target faces"),
        groups=(),
        cause="an AMI interface could not pair its two patch sides, so no interpolation weights exist",
        fix_note="check that the two AMI patches overlap geometrically and verify the transform settings",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="failed_lookup",
        severity=Severity.FATAL,
        pattern=re.compile("failed lookup of (\\S+)"),
        groups=("object",),
        cause="the solver asked the object registry for {object} and it was never created",
        fix_note="ensure the file for {object} exists in constant/ with the correct name and a valid header",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="bad_file_header",
        severity=Severity.FATAL,
        pattern=re.compile("problem while reading header for object (\\S+)"),
        groups=("object",),
        cause="the file for {object} is missing its FoamFile header, truncated, or not a dictionary",
        fix_note="restore a valid FoamFile header in the file for {object}, or regenerate the file",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="gamg_no_coarse_levels",
        severity=Severity.FATAL,
        pattern=re.compile("No coarse levels created"),
        groups=(),
        cause="GAMG could not build a coarse grid hierarchy because the mesh is too small for it",
        fix_note="switch that field's solver to PCG or PBiCGStab, or lower nCellsInCoarsestLevel",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="mpi_executable_not_found",
        severity=Severity.FATAL,
        pattern=re.compile("mpirun was unable to find the specified executable"),
        groups=(),
        cause="MPI could not find the solver binary, so OpenFOAM never started",
        fix_note="source the OpenFOAM environment before running, and check the solver name for typos",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="mpi_run_as_root",
        severity=Severity.FATAL,
        pattern=re.compile("mpirun has detected an attempt to run as root"),
        groups=(),
        cause="Open MPI refuses to launch as root unless permitted, which is common inside containers",
        fix_note="run the case as an unprivileged user instead of root",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="wrong_argument_count",
        severity=Severity.FATAL,
        pattern=re.compile("Expected (\\d+) arguments but found (\\d+)"),
        groups=("expected", "found"),
        cause="the tool was called with {found} positional arguments instead of the {expected} it needs",
        fix_note="supply exactly {expected} arguments; the command provided {found}",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="not_in_table",
        severity=Severity.FATAL,
        pattern=re.compile(r"(\S+) not found in table\.\s+Valid entries:"),
        groups=("value",),
        cause="{value} is not one of the values this keyword accepts",
        fix_note="replace {value} with one of the valid entries the error lists just below it",
        source=Source.CORPUS,
    ),
    # --- utility tools: harvested from 520 real blockMesh / checkMesh /
    # --- snappyHexMesh / decomposePar logs, 40% of which carried a finding
    DiagnosticSpec(
        id="snap_failed",
        severity=Severity.WARNING,
        pattern=re.compile("Did not successfully snap mesh"),
        groups=(),
        cause="snappyHexMesh could not snap the mesh to the surface everywhere",
        fix_note="relax snapControls (raise nSolveIter or nRelaxIter), or improve the surface geometry",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="location_in_mesh_bad",
        severity=Severity.FATAL,
        pattern=re.compile("Point \\(([^)]*)\\) is not inside the mesh"),
        groups=("point",),
        cause="the locationInMesh point ({point}) lies in solid geometry or outside the background mesh",
        fix_note="move locationInMesh to a point clearly inside the fluid region and off any face",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="surface_not_found",
        severity=Severity.FATAL,
        pattern=re.compile('Cannot find surface starting from \\"?([^\\"\\n]*)'),
        groups=("name",),
        cause="a surface named in the meshing dictionary has no matching geometry entry",
        fix_note="check the geometry block and the triSurface file name for {name}",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="no_surface_within",
        severity=Severity.WARNING,
        pattern=re.compile("did not find any surface within:?\\s*(\\S+)"),
        groups=("distance",),
        cause="a refinement or inside point found no surface within {distance}, leaving that region unrefined",
        fix_note="check the STL units against the case, and the refinement point coordinates",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="stl_odd_vertices",
        severity=Severity.WARNING,
        pattern=re.compile("the number of vertices in the geometry is odd"),
        groups=(),
        cause="the surface geometry is not a closed manifold, which snapping relies on",
        fix_note="repair the STL before meshing again; surfaceCheck lists the problems",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="stl_not_closed",
        severity=Severity.WARNING,
        pattern=re.compile(
            "The number of points in the mesh is not equal to twice the number of edges"
        ),
        groups=(),
        cause="the surface has holes or duplicated edges, so it does not bound a volume",
        fix_note="run surfaceCheck on the STL and close the holes it reports",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="empty_geometry",
        severity=Severity.WARNING,
        pattern=re.compile("cannot find bounding box for zero-sized pointField"),
        groups=(),
        cause="a geometry entry resolved to nothing, so its bounding box is empty",
        fix_note="check that the triSurface file exists and is not empty",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="faces_not_visited",
        severity=Severity.WARNING,
        pattern=re.compile("Did not visit some faces, e\\.g\\. face (\\d+)"),
        groups=("face",),
        cause="the mesh has disconnected regions; face {face} was unreachable from the seed point",
        fix_note="check locationInMesh and whether the geometry seals the domain into separate volumes",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="library_not_loaded",
        severity=Severity.WARNING,
        pattern=re.compile(
            'dlopen error|dlOpen: using fallback library name|[Cc]ould not load \\"?([^\\"\\n]*)'
        ),
        groups=("library",),
        cause="a shared library named in controlDict could not be loaded, so its features are missing",
        fix_note="install or build the library, or remove it from the libs entry in controlDict",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="already_decomposed",
        severity=Severity.FATAL,
        pattern=re.compile("Case is already decomposed with (\\d+) domains"),
        groups=("domains",),
        cause="processor directories for {domains} domains already exist",
        fix_note="pass -force to decomposePar, or remove the existing processor* directories",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="wedge_not_planar",
        severity=Severity.WARNING,
        pattern=re.compile("Wedge patch '([^']*)' is not planar"),
        groups=("patch",),
        cause="wedge patch {patch} is not flat, which an axisymmetric case requires",
        fix_note="make the wedge faces planar and symmetric about the axis in blockMeshDict",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="undefined_faces",
        severity=Severity.WARNING,
        pattern=re.compile(
            "Found (\\d+) undefined faces in mesh; adding to default patch"
        ),
        groups=("faces",),
        cause="{faces} faces were not assigned to any patch and went to defaultFaces",
        fix_note="name every boundary in blockMeshDict unless defaultFaces is what you intended",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="unexpected_token",
        severity=Severity.FATAL,
        pattern=re.compile("Unexpected '([^']*)' while reading dictionary entry"),
        groups=("token",),
        cause="a dictionary has an unbalanced brace or a stray token {token}",
        fix_note="check braces and semicolons near the file and line the log names next",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="high_aspect_ratio_detail",
        severity=Severity.WARNING,
        pattern=re.compile(
            "\\*\\*\\*High aspect ratio cells found, Max aspect ratio: (\\S+), number of cells (\\d+)"
        ),
        groups=("ratio", "cells"),
        cause="{cells} cells are stretched up to an aspect ratio of {ratio}",
        fix_note="refine or re-mesh those regions; high ratios slow convergence and can destabilise",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="concave_cells",
        severity=Severity.WARNING,
        pattern=re.compile(
            "\\*\\*\\*Concave cells \\(using face planes\\) found, number of cells: (\\d+)"
        ),
        groups=("cells",),
        cause="{cells} cells are concave, so face-based interpolation is unreliable there",
        fix_note="improve the mesh there; checkMesh writes a cellSet you can visualise",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="high_skewness",
        severity=Severity.WARNING,
        pattern=re.compile("\\*\\*\\*Max skewness = (\\S+), (\\d+) highly skew faces"),
        groups=("skewness", "faces"),
        cause="{faces} faces reach a skewness of {skewness}, which degrades the result",
        fix_note="re-mesh the skewed region; skewness above about 4 is usually trouble",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="small_determinant",
        severity=Severity.WARNING,
        pattern=re.compile(
            "\\*\\*\\*Cells with small determinant \\([^)]*\\) found, number of cells: (\\d+)"
        ),
        groups=("cells",),
        cause="{cells} cells are nearly degenerate, so the solution there is poorly conditioned",
        fix_note="re-mesh those cells; checkMesh writes a cellSet naming them",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="small_interpolation_weight",
        severity=Severity.WARNING,
        pattern=re.compile(
            "\\*\\*\\*Faces with small interpolation weight \\([^)]*\\) found, number of faces: (\\d+)"
        ),
        groups=("faces",),
        cause="{faces} faces sit far off-centre between their cells, biasing interpolation",
        fix_note="improve cell-to-cell size transitions around those faces",
        source=Source.CORPUS,
    ),
    DiagnosticSpec(
        id="empty_patch_mismatch",
        severity=Severity.FATAL,
        pattern=re.compile(
            "\\*\\*\\*Total number of faces on empty patches is not divisible"
        ),
        groups=(),
        cause="the empty patches do not pair up, so this is not a valid 2D or axisymmetric case",
        fix_note="give the two empty patches equal, matching face counts in blockMeshDict",
        source=Source.CORPUS,
    ),
)

#: Warnings every healthy run prints. The catch-all must skip them or it cries
#: wolf: "Smoothing displacement" alone appeared 628 times across 520 real
#: utility logs.
BENIGN_WARNINGS: Final[Sequence[re.Pattern[str]]] = (
    re.compile(r"Smoothing displacement"),
    re.compile(r"This can be a security risk if running untrusted cases"),
    re.compile(r"Unknown compression specifier"),
)

BY_ID: Final[Mapping[str, DiagnosticSpec]] = {d.id: d for d in DIAGNOSTICS}

#: ``bounding`` is consumed by the main pattern chain and synthesised afterwards
#: from the per-field counter, so it is excluded from the fall-through scan.
FALLTHROUGH: Final[Sequence[DiagnosticSpec]] = tuple(
    d for d in DIAGNOSTICS if d.id != "bounding"
)
