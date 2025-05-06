# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_fields.py

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from foamlib import FoamFieldFile

code_p = dedent(
    """
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      p;
    }}

    dimensions    [0 2 -2 0 0 0 0];

    internalField {}

    boundaryField
    {{
    }}
"""
).strip()


def test_p(tmp_path: Path) -> None:
    path = tmp_path / "p"

    path.write_text(code_p.format("uniform 2.0;"))
    field = FoamFieldFile(path)
    assert field.dimensions == FoamFieldFile.DimensionSet(length=2, time=-2)
    assert field.internal_field == 2.0
    assert not field.boundary_field

    path.write_text(
        code_p.format("nonuniform List<scalar>\n3\n(\n    1.0\n    2.0\n    3.0\n);")
    )
    field = FoamFieldFile(path)
    assert field.dimensions == FoamFieldFile.DimensionSet(length=2, time=-2)
    assert np.allclose(field.internal_field, [1.0, 2.0, 3.0])
    assert not field.boundary_field


code_nut = dedent(
    """
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volScalarField;
        object      nut;
    }

    dimensions       [0 2 -1 0 0 0 0];

    internalField    uniform 0;

    boundaryField
    {
        wall
        {
            type     nutkWallFunction;
            value    $internalField;
        }
        #includeEtc    "caseDicts/setConstraintTypes"
    }
"""
)


def test_nut(tmp_path: Path) -> None:
    path = tmp_path / "nut"
    path.write_text(code_nut)
    field = FoamFieldFile(path)

    assert field.dimensions == FoamFieldFile.DimensionSet(length=2, time=-1)
    assert field.internal_field == 0
    assert isinstance(field.boundary_field["wall"], FoamFieldFile.BoundarySubDict)
    assert field.boundary_field["wall"].type == "nutkWallFunction"
    assert field.boundary_field["wall"].value == "$internalField"
    assert field.boundary_field["#includeEtc"] == '"caseDicts/setConstraintTypes"'


code_vector = dedent(
    r"""
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       volVectorField;
        object      U;
    }

    dimensions    [0 1 -1 0 0 0 0];

    internalField   nonuniform List<vector>
    3
    (
        (-1 2 3)
        (-2 4 6)
        (-3 6 9)
    );

    boundaryField
    {
    }
"""
)


def test_vector(tmp_path: Path) -> None:
    path = tmp_path / "U"
    path.write_text(code_vector)
    field = FoamFieldFile(path)

    assert field.dimensions == FoamFieldFile.DimensionSet(length=1, time=-1)
    assert np.allclose(
        field.internal_field,
        [[-1, 2, 3], [-2, 4, 6], [-3, 6, 9]],
    )
    assert not field.boundary_field


code_cells_centers = dedent(
    r"""
    /*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\    /   O peration     | Version:  2212                                  |
    |   \\  /    A nd           | Website:  www.openfoam.com                      |
    |    \\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        arch        "LSB;label=32;scalar=64";
        class       volVectorField;
        location    "0";
        object      C;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      [0 1 0 0 0 0 0];

    internalField   nonuniform List<vector>
    14
    (
    (0.0785398163397 0.0785398163397 0.0785398163396)
    (0.235619449019 0.0785398163398 0.0785398163397)
    (0.392699081699 0.0785398163399 0.07853981634)
    (0.549778714378 0.07853981634 0.0785398163399)
    (0.706858347058 0.07853981634 0.0785398163399)
    (0.863937979737 0.0785398163398 0.0785398163398)
    (1.02101761242 0.0785398163395 0.0785398163397)
    (1.1780972451 0.0785398163397 0.0785398163399)
    (1.33517687778 0.0785398163399 0.07853981634)
    (1.49225651046 0.0785398163397 0.0785398163398)
    (1.64933614314 0.0785398163398 0.0785398163398)
    (1.80641577581 0.0785398163398 0.0785398163399)
    (1.96349540849 0.0785398163397 0.0785398163398)
    (2.12057504117 0.0785398163396 0.0785398163398)
    )
    ;

    boundaryField
    {
        upperBoundary
        {
            type            calculated;
            value           nonuniform List<vector>
    2
    (
    (50 8000 0.005)
    (150 8000 0.005)
    )
    ;
        }
        lowerBoundary
        {
            type            cyclic;
        }
        leftBoundary
        {
            type            cyclic;
        }
        rightBoundary
        {
            type            cyclic;
        }
        frontBoundary
        {
            type            cyclic;
        }
        backBoundary
        {
            type            cyclic;
        }
    }


    // ************************************************************************* //
"""
)

code_cx = dedent(
    r"""
    /*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\    /   O peration     | Version:  2212                                  |
    |   \\  /    A nd           | Website:  www.openfoam.com                      |
    |    \\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        arch        "LSB;label=32;scalar=64";
        class       volScalarField;
        location    "0";
        object      Cx;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      [0 1 0 0 0 0 0];

    internalField   nonuniform List<scalar>
    14
    (
    0.0785398163397
    0.235619449019
    0.392699081699
    0.549778714378
    0.706858347058
    0.863937979737
    1.02101761242
    1.1780972451
    1.33517687778
    1.49225651046
    1.64933614314
    1.80641577581
    1.96349540849
    2.12057504117
    )
    ;

    boundaryField
    {
        upperBoundary
        {
            type            cyclic;
        }
        lowerBoundary
        {
            type            cyclic;
        }
        leftBoundary
        {
            type            cyclic;
        }
        rightBoundary
        {
            type            cyclic;
        }
        frontBoundary
        {
            type            cyclic;
        }
        backBoundary
        {
            type            cyclic;
        }
    }


    // ************************************************************************* //
"""
)


def test_cell_centers(tmp_path: Path) -> None:
    """Data obtained with `postProcess -func writeCellCentres`

    https://www.openfoam.com/documentation/guides/latest/doc/guide-fos-field-writeCellCentres.html

    """
    path = tmp_path / "C"
    path.write_text(code_cells_centers)
    field_c = FoamFieldFile(path)

    path = tmp_path / "Cx"
    path.write_text(code_cx)
    field_cx = FoamFieldFile(path)

    assert field_c.dimensions == FoamFieldFile.DimensionSet(length=1)
    assert field_cx.dimensions == FoamFieldFile.DimensionSet(length=1)

    assert isinstance(field_c.internal_field, np.ndarray)
    assert field_c.internal_field[:, 0] == pytest.approx(field_cx.internal_field)
