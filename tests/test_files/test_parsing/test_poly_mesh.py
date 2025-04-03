# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_polymesh.py


from pathlib import Path

from foamlib import FoamFile

contents = r"""
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2206                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


10
(
(0 0 0)
(0.15707963268 0 0)
(0.314159265359 0 0)
(0.471238898038 0 0)
(0.628318530718 0 0)
(0 0 0)
(0.15707963268 0 0)
(0.314159265359 0 0)
(0.471238898038 0 0)
(0.628318530718 0 0)
)

// ************************************************************************* //
"""


def test_get_cells_coords(tmp_path: Path) -> None:
    path = tmp_path / "points"
    path.write_text(contents)

    points = FoamFile(path)

    assert points[None][0] == [0, 0, 0]
    assert points[None][1] == [0.15707963268, 0, 0]
    assert points[None][2] == [0.314159265359, 0, 0]
    assert points[None][3] == [0.471238898038, 0, 0]
    assert points[None][4] == [0.628318530718, 0, 0]
    assert points[None][5] == [0, 0, 0]
    assert points[None][6] == [0.15707963268, 0, 0]
    assert points[None][7] == [0.314159265359, 0, 0]
    assert points[None][8] == [0.471238898038, 0, 0]
    assert points[None][9] == [0.628318530718, 0, 0]

    assert len(points[None]) == 10

    assert list(points) == [None]
    assert "FoamFile" in points
