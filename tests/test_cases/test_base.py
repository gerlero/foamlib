from pathlib import Path

from foamlib import FoamCaseBase, FoamFieldFile


def test_create_from_scratch(tmp_path: Path) -> None:
    case = FoamCaseBase(tmp_path / "case")
    assert not case.path.exists()

    assert not case[0.0].path.exists()
    assert not case[0.0]["U"].path.exists()
    case[0.0]["U"] = {
        "dimensions": [0, 1, -1, 0, 0, 0, 0],
        "internalField": [1.0, 2.0, 3.0],
        "boundaryField": {
            "movingWall": {"type": "fixedValue", "value": [1.0, 2.0, 3.0]}
        },
    }
    assert case[0.0]["U"].path.is_file()
    assert isinstance(case[0.0]["U"], FoamFieldFile)
    assert len(case) == 1
    assert 0.0 in case
    assert "U" in case[0.0]
    assert case[0.0]
    assert len(case[0.0]) == 1

    del case[0.0]["U"]
    assert len(case[0.0]) == 0
    assert not case[0.0]
    assert len(case) == 1
    assert 0.0 in case
    assert case[0.0].path.is_dir()

    del case[:]
    assert len(case) == 0
    assert not case
