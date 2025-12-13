from foamlib import FoamFile

CONTENTS = """7
(
    xMin
    {
        type            symmetry;
        inGroups        1(symmetry);
        nFaces          1036;
        startFace       34815683;
    }
    xMax
    {
        type            symmetry;
        inGroups        1(symmetry);
        nFaces          1036;
        startFace       34816719;
    }
    yMin
    {
        type            symmetry;
        inGroups        1(symmetry);
        nFaces          1036;
        startFace       34817755;
    }
    yMax
    {
        type            symmetry;
        inGroups        1(symmetry);
        nFaces          1036;
        startFace       34818791;
    }
    zMin
    {
        type            patch;
        nFaces          784;
        startFace       34819827;
    }
    zMax
    {
        type            patch;
        nFaces          784;
        startFace       34820611;
    }
    another-patch
    {
        type            wall;
        inGroups        1(wall);
        nFaces          105416;
        startFace       34821395;
    }
)"""


def test_issue_605() -> None:
    FoamFile.loads(CONTENTS)
