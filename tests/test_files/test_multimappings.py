from pathlib import Path

import pytest

from foamlib import FoamFile


@pytest.fixture
def file(tmp_path: Path) -> FoamFile:
    ret = FoamFile(tmp_path / "testDict")
    with ret:
        ret.add("#include", '"first"')
        ret.add("#include", '"second"')
        ret["value"] = 1
        ret["sub"] = {}

        sub = ret["sub"]
        assert isinstance(sub, FoamFile.SubDict)
        sub.add("#include", '"third"')
        sub.add("#include", '"fourth"')
        sub["value"] = 2
    return ret


def test_file_views_with_duplicate_keys(file: FoamFile) -> None:
    assert list(file) == ["#include", "value", "sub"]
    assert list(file.keys()) == ["#include", "value", "sub"]
    assert len(file.keys()) == 3

    items = list(file.items())
    values = list(file.values())
    assert [key for key, _ in items] == ["#include", "#include", "value", "sub"]
    assert len(file) == len(file.items()) == len(file.values()) == 4
    assert len(items) == len(values) == 4
    assert values[:3] == [value for _, value in items[:3]]
    assert isinstance(values[-1], FoamFile.SubDict)


def test_file_views_include_header(file: FoamFile) -> None:
    assert list(file.keys(include_header=True)) == [
        "FoamFile",
        "#include",
        "value",
        "sub",
    ]
    assert [key for key, _ in file.items(include_header=True)] == [
        "FoamFile",
        "#include",
        "#include",
        "value",
        "sub",
    ]
    assert len(file.keys(include_header=True)) == 4
    assert len(file.items(include_header=True)) == 5
    assert len(file.values(include_header=True)) == 5
    assert isinstance(next(iter(file.values(include_header=True))), FoamFile.SubDict)


def test_subdict_views_with_duplicate_keys(file: FoamFile) -> None:
    sub = file["sub"]
    assert isinstance(sub, FoamFile.SubDict)

    assert list(sub) == ["#include", "value"]
    assert list(sub.keys()) == ["#include", "value"]
    assert len(sub.keys()) == 2

    items = list(sub.items())
    values = list(sub.values())
    assert [key for key, _ in items] == ["#include", "#include", "value"]
    assert len(sub) == len(sub.items()) == len(sub.values()) == 3
    assert len(items) == len(values) == 3
    assert values == [value for _, value in items]


@pytest.mark.parametrize("nested", [False, True])
def test_update_with_duplicate_keys(tmp_path: Path, nested: bool) -> None:
    file = FoamFile(tmp_path / "testDict")
    mapping: FoamFile | FoamFile.SubDict
    if nested:
        file["sub"] = {}
        sub = file["sub"]
        assert isinstance(sub, FoamFile.SubDict)
        mapping = sub
    else:
        mapping = file

    with file:
        mapping.add("#include", '"first"')
        mapping.add("#include", '"second"')
        mapping["target"] = 1

    mapping.update({"target": 2})

    assert mapping["target"] == 2
    assert len(mapping.getall("#include")) == 2
    assert len(mapping) == 3
    assert [key for key, _ in mapping.items()] == ["#include", "#include", "target"]
    assert list(mapping.getall("#include")) == ['"first"', '"second"']
