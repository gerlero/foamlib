import numpy as np
from multicollections import MultiDict

from foamlib import dumps


def test_dumps() -> None:
    assert dumps(1.0, ensure_header=False) == b"1.0"
    assert dumps(1.0) == b"FoamFile {version 2.0; format ascii; class dictionary;} 1.0"
    assert (
        dumps({"a": "b", "c": "d", "n": False, "y": True})
        == b"FoamFile {version 2.0; format ascii; class dictionary;} a b; c d; n no; y yes;"
    )
    assert (
        dumps({"internalField": [[1, 2, 3], [4, 5, 6]]})
        == b"FoamFile {version 2.0; format ascii; class volVectorField;} internalField nonuniform List<vector> 2((1.0 2.0 3.0) (4.0 5.0 6.0));"
    )
    assert (
        dumps([1, 2, 3, 4, 5, 6])
        == b"FoamFile {version 2.0; format ascii; class dictionary;} (1 2 3 4 5 6)"
    )
    assert dumps([1, 2, 3], ensure_header=False) == b"(1 2 3)"
    assert dumps([1.0, 2, 3], ensure_header=False) == b"(1.0 2.0 3.0)"
    assert (
        dumps([[1, 2, 3], [4, 5.0, 6]], ensure_header=False)
        == b"((1.0 2.0 3.0) (4.0 5.0 6.0))"
    )
    assert (
        dumps(
            {
                "FoamFile": {"format": "binary"},
                None: np.array([1, 2, 3], dtype=np.int32),
            },
            ensure_header=False,
        )
        == b"FoamFile {format binary;} 3(\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00)"
    )
    assert (
        dumps({"#include": "$FOAM_CASE/simControls"}, ensure_header=False)
        == b"\n#include $FOAM_CASE/simControls\n"
    )
    assert (
        dumps([[1, 2, 3, 4], [5, 6, 7, 8]], ensure_header=False)
        == b"(4(1 2 3 4) 4(5 6 7 8))"
    )
    assert (
        dumps([[1, 2, 3], [4, 5, 6, 7]], ensure_header=False)
        == b"(3(1 2 3) 4(4 5 6 7))"
    )
    assert dumps([[1, 2, 3], [4, 5, 6]], ensure_header=False) == b"(3(1 2 3) 3(4 5 6))"
    indices = np.array([0, 3, 7], dtype=np.int32)
    values = np.array(
        [904040, 904479, 924424, 3516631, 3516634, 3516633, 3516632], dtype=np.int32
    )
    assert (
        dumps(
            {"FoamFile": {"format": "binary"}, None: (indices, values)},
            ensure_header=False,
        )
        == b"FoamFile {format binary;} 3(\x00\x00\x00\x00\x03\x00\x00\x00\x07\x00\x00\x00) 7(h\xcb\r\x00\x1f\xcd\r\x00\x08\x1b\x0e\x00\xd7\xa85\x00\xda\xa85\x00\xd9\xa85\x00\xd8\xa85\x00)"
    )
    assert (
        dumps(
            MultiDict([("#includeFunc", '"func1"'), ("#includeFunc", '"func2"')]),
            ensure_header=False,
        )
        == b'\n#includeFunc "func1"\n \n#includeFunc "func2"\n'
    )
    assert dumps({"keyword": None}, ensure_header=False) == b"keyword;"
