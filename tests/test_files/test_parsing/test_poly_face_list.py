from pathlib import Path

import numpy as np
from foamlib import FoamFile

faces_contents = r"""
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
    class       faceList;
    location    "constant/polyMesh";
    object      faces;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

3
(
3(0 1 2)
4(3 4 5 6)
5(7 8 9 10 11)
)

// ************************************************************************* //
"""


def test_parse_poly_faces(tmp_path: Path) -> None:
    """Test that ascii faceList with triangles, quads, and pentagons is parsed correctly."""
    path = tmp_path / "faces"
    path.write_text(faces_contents)

    file = FoamFile(path)
    faces = file[None]

    assert len(faces) == 3
    assert np.array_equal(faces[0], [0, 1, 2])
    assert np.array_equal(faces[1], [3, 4, 5, 6])
    assert np.array_equal(faces[2], [7, 8, 9, 10, 11])


float_list_list_contents = r"""
3
(
2(0.1 0.2)
3(0.3 0.4 0.5)
1(0.6)
)
"""


def test_parse_float_list_list(tmp_path: Path) -> None:
    """Test that a standalone ascii numeric list-of-lists with float values is parsed correctly."""
    path = tmp_path / "floats"
    path.write_text(float_list_list_contents)

    file = FoamFile(path)
    data = file[None]

    assert len(data) == 3
    assert np.allclose(data[0], [0.1, 0.2])
    assert np.allclose(data[1], [0.3, 0.4, 0.5])
    assert np.allclose(data[2], [0.6])


commented_faces_contents = r"""
3
(
3(0 1 2) // triangle
4 /* quad */ (3 4 5 6)
5(
  7 // comment inside
  8
  9
  10
  11
)
)
"""


def test_parse_commented_faces(tmp_path: Path) -> None:
    """Test that ascii faceList with inline comments is parsed correctly."""
    path = tmp_path / "faces_commented"
    path.write_text(commented_faces_contents)

    file = FoamFile(path)
    faces = file[None]

    assert len(faces) == 3
    assert np.array_equal(faces[0], [0, 1, 2])
    assert np.array_equal(faces[1], [3, 4, 5, 6])
    assert np.array_equal(faces[2], [7, 8, 9, 10, 11])
