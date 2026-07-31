"""
Dimension-set exponent-count parity with OpenFOAM.

OpenFOAM's ``dimensionSet`` file grammar accepts exactly 5 or 7 numeric exponents
(``dimensionSetIO.C``): 1-4 exponents hit ``FatalIOError`` reading ``]`` as a scalar,
and 6 exponents read ``]`` as the luminous-intensity exponent and then fail the
end-of-set check. The ``DimensionSet`` constructor mirrors the 7-argument C++ form
with ``current`` and ``luminous_intensity`` defaulting to 0, so 5- and 6-argument
calls are valid and pad the missing trailing exponents.
"""

import numpy as np
import pytest

from foamlib import DimensionSet, FoamFile, FoamFileDecodeError
from foamlib._files._parsing import ParsedFile

_READ_REJECT = [
    b"[1]",
    b"[1 2]",
    b"[1 2 3]",
    b"[1 2 3 4]",
    b"[0 1 -2]",
    b"[1 2 3 4 5 6]",
    b"[0 1 -2 0 0 0 0 1]",
]

_READ_ACCEPT = [
    (b"[]", DimensionSet()),
    (b"[1 2 3 4 5]", DimensionSet(mass=1, length=2, time=3, temperature=4, moles=5)),
    (
        b"[1 2 3 4 5 6 7]",
        DimensionSet(
            mass=1,
            length=2,
            time=3,
            temperature=4,
            moles=5,
            current=6,
            luminous_intensity=7,
        ),
    ),
    (b"[0 1 -2 0 0 0 0]", DimensionSet(length=1, time=-2)),
    (b"[0 0.5 0 0 0 0 0]", DimensionSet(length=0.5)),
    (b"[mass]", DimensionSet(mass=1)),
]


@pytest.mark.parametrize("contents", _READ_REJECT)
def test_read_rejects_wrong_exponent_count(contents: bytes) -> None:
    with pytest.raises(FoamFileDecodeError):
        FoamFile.loads(b"dimensions " + contents + b";")


@pytest.mark.parametrize(("contents", "expected"), _READ_ACCEPT)
def test_read_accepts_valid_dimension_sets(
    contents: bytes, expected: DimensionSet
) -> None:
    assert ParsedFile(b"dimensions " + contents + b";")[("dimensions",)] == expected


@pytest.mark.parametrize("n", [5, 6, 7])
def test_constructor_pads_optional_trailing_exponents(n: int) -> None:
    dims = DimensionSet(*range(1, n + 1))
    assert tuple(dims) == tuple(range(1, n + 1)) + (0,) * (7 - n)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_constructor_rejects_too_few_or_too_many_arguments(n: int) -> None:
    with pytest.raises(TypeError, match="5 to 7"):
        DimensionSet(*range(1, n + 1))


@pytest.mark.parametrize("n", [5, 6, 7])
def test_dumps_dimension_set_round_trips(n: int) -> None:
    data = {"dimensions": list(range(1, n + 1))}
    dumped = FoamFile.dumps(data)  # ty: ignore[invalid-argument-type]
    exponents = " ".join([str(i) for i in range(1, n + 1)] + ["0"] * (7 - n))
    assert dumped.endswith(f"dimensions [{exponents}];".encode())
    assert ParsedFile(dumped)[("dimensions",)] == DimensionSet(*range(1, n + 1))


def test_dimension_set_arithmetic_after_padding() -> None:
    assert DimensionSet(1, 2, 3, 4, 5) * DimensionSet(1, 1, 1, 1, 1) == DimensionSet(
        mass=2, length=3, time=4, temperature=5, moles=6
    )
    assert np.array_equal(
        np.array(DimensionSet(1, 2, 3, 4, 5, 6)), [1, 2, 3, 4, 5, 6, 0]
    )
