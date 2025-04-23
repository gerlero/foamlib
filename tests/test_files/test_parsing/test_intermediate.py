# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_parser.py

import numpy as np
import pytest
from foamlib import FoamFile
from foamlib._files._parsing import Parsed


def test_var_simple() -> None:
    assert Parsed(b"a  b;")[("a",)] == "b"


def test_var_quoted_string() -> None:
    assert (
        Parsed(b"""
        laplacianSchemes
        {
            default    "Gauss linear corrected";
        }
    """)[("laplacianSchemes", "default")]
        == '"Gauss linear corrected"'
    )


def test_var_multiple() -> None:
    parsed = Parsed(b"""
        a    b;

        c    d;
    """)
    assert parsed[("a",)] == "b"
    assert parsed[("c",)] == "d"


@pytest.mark.xfail(reason="Not currently supported")
def test_strange_names() -> None:
    parsed = Parsed(b"""
        "(U|k|epsilon|R)Final"
        {
            $U;
            tolerance    1e-07;
            relTol       0;
        }

        thermalPhaseChange:dmdtf  1.0;

        thermo:rho
        {
            solver    PCG;
        }

        alpha.water
        {
            solver    PCG;
        }

    """)
    assert parsed[("(U|k|epsilon|R)Final", "$U")] == ""
    assert parsed[("thermalPhaseChange:dmdtf",)] == 1.0
    assert parsed[("thermo:rho", "solver")] == "PCG"
    assert parsed[("alpha.water", "solver")] == "PCG"


def test_list_simple() -> None:
    assert Parsed(b"""
        faces
        (
            (1 5 4 0)
            (2 3 4 5)
        );
    """)[("faces",)] == [
        [1, 5, 4, 0],
        [2, 3, 4, 5],
    ]


def test_list_assignment() -> None:
    assert Parsed(b"""
        faces
        (
            1
            5
            4
            0
        );
    """)[("faces",)] == [1, 5, 4, 0]


def test_dict_simple() -> None:
    parsed = Parsed(b"""
        my_dict
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            location    "system";
            object      controlDict;
        }
    """)
    assert parsed[("my_dict", "version")] == 2.0
    assert parsed[("my_dict", "format")] == "ascii"
    assert parsed[("my_dict", "location")] == '"system"'


def test_dict_nested() -> None:
    parsed = Parsed(b"""
        my_nested_dict
        {
            p
            {
                solver            PCG;
                preconditioner    DIC;
                tolerance         1e-06;
                relTol            0.05;
            }
            U
            {
                solver       smoothSolver;
                smoother     symGaussSeidel;
                tolerance    1e-05;
                relTol       0;
            }
        }
    """)
    assert parsed[("my_nested_dict", "p", "solver")] == "PCG"
    assert parsed[("my_nested_dict", "U", "tolerance")] == 1e-05


def test_dict_with_list() -> None:
    parsed = Parsed(b"""
        PISO
        {
            nCorrectors                 2;
            nNonOrthogonalCorrectors    1;
            pRefPoint                   (0 0 0);
            pRefValue                   0;
        }
    """)
    assert parsed[("PISO", "pRefPoint")] == [0, 0, 0]


def test_list_with_dict() -> None:
    assert Parsed(b"""
        boundary
        (
            upperBoundary
            {
                type              cyclic;
                neighbourPatch    lowerBoundary;
                faces
                (
                    (3 7 6 2)
                );
            }
        );
    """)[("boundary",)] == [
        (
            "upperBoundary",
            {
                "type": "cyclic",
                "neighbourPatch": "lowerBoundary",
                "faces": [
                    [3, 7, 6, 2],
                ],
            },
        ),
    ]


def test_list_with_str() -> None:
    assert Parsed(b"""
        blocks
        (
            hex (0 1 2 3 4 5 6 7) (40 40 40) simpleGrading (1 1 1)
        );
    """)[("blocks",)] == [
        "hex",
        [0, 1, 2, 3, 4, 5, 6, 7],
        [40, 40, 40],
        "simpleGrading",
        [1, 1, 1],
    ]


def test_file_simple() -> None:
    parsed = Parsed(b"""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            object      blockMeshDict;
        }

        a  b;

        c  d;
    """)
    assert parsed[("a",)] == "b"
    assert parsed[("c",)] == "d"
    assert parsed[("FoamFile", "version")] == 2.0
    assert parsed[("FoamFile", "format")] == "ascii"
    assert parsed[("FoamFile", "class")] == "dictionary"
    assert parsed[("FoamFile", "object")] == "blockMeshDict"


def test_file() -> None:
    parsed = Parsed(b"""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      p;
        }

        a    1;

        b    2;

        faces
        (
            (1 5 4 0)
            (2 3 4 5)
        );

        my_dict
        {
            a    1;
        }
    """)
    assert parsed[("a",)] == 1
    assert parsed[("b",)] == 2
    assert parsed[("faces",)] == [
        [1, 5, 4, 0],
        [2, 3, 4, 5],
    ]
    assert parsed[("my_dict", "a")] == 1


def test_directive() -> None:
    parsed = Parsed(b"""
        FoamFile
        {
            version     2.0;
        }

        #include  "initialConditions"
    """)
    assert parsed[("FoamFile", "version")] == 2.0
    assert parsed[("#include",)] == '"initialConditions"'


@pytest.mark.xfail(reason="Not currently supported")
def test_directives_in_dict() -> None:
    Parsed(b"""
        functions
        {
            #includeFunc fieldAverage(cylindrical(U))
            #includeFunc Qdot
            #includeFunc components(U)
            #includeFunc Qdot(region=gas)
            #includeFunc residuals(region = shell, p_rgh, U, h)
            #includeFunc residuals(region = tube, p_rgh, U, h)
            #includeFunc patchAverage
            (
                funcName=cylinderT,
                region=fluid,
                patch=fluid_to_solid,
                field=T
            )
            #includeFunc streamlinesLine(funcName=streamlines, start=(0 0.5 0), end=(9 0.5 0), nPoints=24, U)
            #includeFunc streamlinesLine
            (
                funcName=streamlines,
                start=(-0.0205 0.001 0.00001),
                end=(-0.0205 0.0251 0.00001),
                nPoints=10,
                fields=(p k U)
            )
            #includeFunc writeObjects(kEpsilon:G)
            #includeFunc fieldAverage(U, p, alpha.vapour)
            #includeFunc writeObjects
            (
                d.particles,
                a.particles,
                phaseTransfer:dmidtf.TiO2.particlesAndVapor,
                phaseTransfer:dmidtf.TiO2_s.particlesAndVapor
            )
            #includeFunc  graphUniform
            (
                funcName=graph,
                start=(0 0 0.89),
                end=(0.025 0 0.89),
                nPoints=100,
                fields=
                (
                    alpha.air1
                    alpha.air2
                    alpha.bubbles
                    liftForce.water
                    wallLubricationForce.water
                    turbulentDispersionForce.water
                )
            )
        }
    """)


@pytest.mark.xfail(reason="Not currently supported")
def test_code() -> None:
    Parsed(b"""
        code_name
        #{
            -I$(LIB_SRC)/finiteVolume/lnInclude \
            -I$(LIB_SRC)/meshTools/lnInclude
        #};
    """)


def test_macro() -> None:
    parsed = Parsed(b"""
        FoamFile
        {
            version     2.0;
        }

        relTol            $p;

        Phi
        {
            $p;
        }

        p_rbghFinal
        {
            $p_rbgh;
            tolerance    1e-08;
            relTol       0;
        }

        relaxationFactors  $relaxationFactors-SIMPLE;
    """)
    assert parsed[("relTol",)] == "$p"
    assert parsed[("Phi", "$p")] == ""
    assert parsed[("p_rbghFinal", "$p_rbgh")] == ""
    assert parsed[("p_rbghFinal", "tolerance")] == 1e-08
    assert parsed[("p_rbghFinal", "relTol")] == 0
    assert parsed[("relaxationFactors",)] == "$relaxationFactors-SIMPLE"


def test_empty_dict() -> None:
    Parsed(b"""
        solvers
        {
        }
        relaxationFactors
        {}
    """)


def test_dict_isolated_key() -> None:
    parsed = Parsed(b"""
        cache
        {
            grad(U);
        }
    """)
    assert parsed[("cache", "grad(U)")] == ""


def test_dimension_set() -> None:
    parsed = Parsed(b"""
        dimensions    [0 2 -1 0 0 0 0];

        nu            [0 2 -1 0 0 0 0] 1e-05;

        nu1           nu [0 2 -1 0 0 0 0] 1e-06;

        SIMPLE
        {
            rhoMin    rhoMin [1 -3 0 0 0 0 0] 0.3;
        }
    """)
    assert parsed[("dimensions",)] == FoamFile.DimensionSet(length=2, time=-1)
    assert parsed[("nu",)].dimensions == FoamFile.DimensionSet(length=2, time=-1)
    assert parsed[("nu1",)].dimensions == FoamFile.DimensionSet(length=2, time=-1)
    assert parsed[("SIMPLE", "rhoMin")].dimensions == FoamFile.DimensionSet(
        mass=1, length=-3
    )


def test_named_values() -> None:
    parsed = Parsed(b"""
        a     b;

        ft    limitedLinear01 1;
    """)
    assert parsed[("a",)] == "b"
    assert parsed[("ft",)] == ("limitedLinear01", 1)


@pytest.mark.xfail(reason="Not currently supported")
def test_macro_ugly() -> None:
    parsed = Parsed(b"""
        relaxationFactors
        {
            ${_${FOAM_EXECUTABLE}};
        }
    """)
    assert parsed[("relaxationFactors", "${_${FOAM_EXECUTABLE}}")] == ""


def test_list_on_1_line() -> None:
    parsed = Parsed(b"""
        libs            (overset rigidBodyDynamics);

        functions
        {
            minMax1
            {
                libs            (fieldFunctionObjects);
                type            fieldMinMax;
                fields          (U p);
            }
        }
    """)
    assert parsed[("libs",)] == ["overset", "rigidBodyDynamics"]
    assert parsed[("functions", "minMax1", "libs")] == ["fieldFunctionObjects"]
    assert parsed[("functions", "minMax1", "type")] == "fieldMinMax"
    assert parsed[("functions", "minMax1", "fields")] == ["U", "p"]


def test_double_value() -> None:
    parsed = Parsed(b"""
        FoamFile
        {
            format    ascii;
            object    controlDict.1st;
        }
    """)
    assert parsed[("FoamFile", "format")] == "ascii"
    assert parsed[("FoamFile", "object")] == "controlDict.1st"


def test_for_blockmesh() -> None:
    parsed = Parsed(b"""
        negHalfWidth    #neg $halfWidth;

        blocks
        (
            hex (4 6 14 12 0 2 10 8) (1 $upstreamCells $cylinderBoxCells) $expandBlock
        );
    """)
    assert parsed[("negHalfWidth",)] == ("#neg", "$halfWidth")
    assert parsed[("blocks",)] == [
        "hex",
        [4, 6, 14, 12, 0, 2, 10, 8],
        [1, "$upstreamCells", "$cylinderBoxCells"],
        "$expandBlock",
    ]


def test_for_u() -> None:
    parsed = Parsed(b"""
        internalField  uniform $include/caseSettings!internalField/U;
    """)
    assert parsed[("internalField",)] == (
        "uniform",
        "$include/caseSettings!internalField/U",
    )


def test_blocks() -> None:
    parsed = Parsed(b"""
        blocks
        (
            hex (0 1 2 3 4 5 6 7) inletChannel (40 1 64) simpleGrading (1 1 1)
            hex (4 5 6 7 8 9 10 11 12) inletChannel (40 1 16) simpleGrading (1 1 1)
            hex (12 13 14 15 16 17 18 19) (96 1 8) simpleGrading (1 1 1)
            hex (16 17 18 19 20 21 22 23) (96 1 72) simpleGrading (1 1 1)
        );
    """)
    assert parsed[("blocks",)] == [
        "hex",
        [0, 1, 2, 3, 4, 5, 6, 7],
        "inletChannel",
        [40, 1, 64],
        "simpleGrading",
        [1, 1, 1],
        "hex",
        [4, 5, 6, 7, 8, 9, 10, 11, 12],
        "inletChannel",
        [40, 1, 16],
        "simpleGrading",
        [1, 1, 1],
        "hex",
        [12, 13, 14, 15, 16, 17, 18, 19],
        [96, 1, 8],
        "simpleGrading",
        [1, 1, 1],
        "hex",
        [16, 17, 18, 19, 20, 21, 22, 23],
        [96, 1, 72],
        "simpleGrading",
        [1, 1, 1],
    ]


@pytest.mark.xfail(reason="Not currently supported")
def test_macro_signed() -> None:
    parsed = Parsed(b"""
        vertices
        (
            ($x0 $y0 -$w2)
            (0 -$h2 -$w2)
            (0 $h2 -$w2)
            ($x1 $y1 -$w2)
        );
    """)
    assert parsed[("vertices",)] == [
        ["$x0", "$y0", "-$w2"],
        [0, "-$h2", "-$w2"],
        [0, "$h2", "-$w2"],
        ["$x1", "$y1", "-$w2"],
    ]


@pytest.mark.xfail(reason="Should fail")
def test_list_numbered() -> None:
    with pytest.raises(Exception, match="Expected"):
        Parsed(b"""
            internalField nonuniform
            List<vector>
            4096
            (
                (-0.0376011 0.020584 -0.0051027)
                (-0.0262359 0.0149309 -0.0048244)
                (-0.0141003 0.00810973 -0.00427023)
            );
        """)


def test_list_numbered_u() -> None:
    with pytest.raises(Exception, match="Expected"):
        Parsed(b"""
            70
            (
                (5.74803 0 0)
                (5.74803 0 0)
                (11.3009 0 0)
                (13.4518 0 0)
                (13.4518 0 0)
                (14.0472 0 0)
            );
        """)


def test_colon_double_name() -> None:
    parsed = Parsed(b"""
        DebugSwitches
        {
            compressible::alphatWallBoilingWallFunction                 0;
            compressible::turbulentTemperatureTwoPhaseRadCoupledMixed   0;
        }
    """)
    assert parsed[("DebugSwitches", "compressible::alphatWallBoilingWallFunction")] == 0
    assert (
        parsed[
            (
                "DebugSwitches",
                "compressible::turbulentTemperatureTwoPhaseRadCoupledMixed",
            )
        ]
        == 0
    )


def test_list_edges() -> None:
    parsed = Parsed(b"""
        edges
        (
            spline 1 2 ((0.6 0.0124 0.0) (0.7 0.0395 0.0) (0.8 0.0724 0.0) (0.9 0.132 0.0) (1 0.172 0.0) (1.1 0.132 0.0) (1.2 0.0724 0.0) (1.3 0.0395 0.0) (1.4 0.0124 0.0))
            spline 6 5 ((0.6 0.0124 0.05) (0.7 0.0395 0.05) (0.8 0.0724 0.05) (0.9 0.132 0.05) (1 0.172 0.05) (1.1 0.132 0.05) (1.2 0.0724 0.05) (1.3 0.0395 0.05) (1.4 0.0124 0.05))
        );
    """)
    assert parsed[("edges",)] == [
        "spline",
        1,
        2,
        [
            [0.6, 0.0124, 0.0],
            [0.7, 0.0395, 0.0],
            [0.8, 0.0724, 0.0],
            [0.9, 0.132, 0.0],
            [1, 0.172, 0.0],
            [1.1, 0.132, 0.0],
            [1.2, 0.0724, 0.0],
            [1.3, 0.0395, 0.0],
            [1.4, 0.0124, 0.0],
        ],
        "spline",
        6,
        5,
        [
            [0.6, 0.0124, 0.05],
            [0.7, 0.0395, 0.05],
            [0.8, 0.0724, 0.05],
            [0.9, 0.132, 0.05],
            [1, 0.172, 0.05],
            [1.1, 0.132, 0.05],
            [1.2, 0.0724, 0.05],
            [1.3, 0.0395, 0.05],
            [1.4, 0.0124, 0.05],
        ],
    ]


def test_list_edges_arcs() -> None:
    parsed = Parsed(b"""
        edges
        (
            arc 0 5 origin (0 0 0)
            arc 5 10 origin (0 0 0)
        );
    """)
    assert parsed[("edges",)] == [
        "arc",
        0,
        5,
        "origin",
        [0, 0, 0],
        "arc",
        5,
        10,
        "origin",
        [0, 0, 0],
    ]


def test_list_blocks() -> None:
    parsed = Parsed(b"""
        blocks
        (
            hex (0 1 9 8 7 6 14 15) (50 100 1) simpleGrading (1 ((0.1 0.25 41.9) (0.9 0.75 1)) 1)
            hex (1 2 10 9 6 5 13 14) (50 100 1) simpleGrading (1 ((0.1 0.25 41.9) (0.9 0.75 1)) 1)
            hex (2 3 11 10 5 4 12 13) (225 100 1) simpleGrading (1 ((0.1 0.25 41.9) (0.9 0.75 1)) 1)
        );
    """)
    assert parsed[("blocks",)] == [
        "hex",
        [0, 1, 9, 8, 7, 6, 14, 15],
        [50, 100, 1],
        "simpleGrading",
        [1, [[0.1, 0.25, 41.9], [0.9, 0.75, 1]], 1],
        "hex",
        [1, 2, 10, 9, 6, 5, 13, 14],
        [50, 100, 1],
        "simpleGrading",
        [1, [[0.1, 0.25, 41.9], [0.9, 0.75, 1]], 1],
        "hex",
        [2, 3, 11, 10, 5, 4, 12, 13],
        [225, 100, 1],
        "simpleGrading",
        [1, [[0.1, 0.25, 41.9], [0.9, 0.75, 1]], 1],
    ]


@pytest.mark.xfail(reason="Not currently supported")
def test_code_stream() -> None:
    Parsed(b"""
        internalField  #codeStream
        {
            codeInclude
            #{
                #include "fvCFD.H"
            #};
            codeOptions
            #{
                -I$(LIB_SRC)/finiteVolume/lnInclude \
                -I$(LIB_SRC)/meshTools/lnInclude
            #};
            codeLibs
            #{
                -lmeshTools \
                -lfiniteVolume
            #};
            code
            #{
                const IOdictionary& d = static_cast<const IOdictionary&>(dict);
                const fvMesh& mesh = refCast<const fvMesh>(d.db());
                scalarField p(mesh.nCells(), 0.);
                forAll(p, i)
                {
                    const scalar x = mesh.C()[i][0];
                    const scalar y = mesh.C()[i][1];
                    const scalar z = mesh.C()[i][2];
                    p[i]=-0.0625*(Foam::cos(2*x) + Foam::cos(2*y))*Foam::cos(2*z+2);
                }
                p.writeEntry("",os);
            #};
        };
    """)


def test_list_uniform() -> None:
    parsed = Parsed(b"""
            a    1;

            internalField uniform
            (
                0.1
                0
                0
            );
        """)
    assert parsed[("a",)] == 1
    assert np.allclose(parsed[("internalField",)], [0.1, 0, 0])
