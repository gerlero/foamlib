"""Tests for binary ScalarList parsing (issue: sporadic parsing errors with binary format)."""

import numpy as np
import pytest

from foamlib._files._parsing._parser import parse_located
from foamlib._files._serialization import dumps


@pytest.mark.parametrize("seed", [141, 539, 566, 685, 728, 773, 777, 907, 953, 992])
def test_binary_scalar_list_round_trip(seed: int) -> None:
    """Test that binary float64 ScalarList data can be serialized and parsed correctly."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    data = rng.random(n)

    content = dumps({"FoamFile": {"format": "binary"}, None: data})
    result = parse_located(content)
    parsed = result[()].data

    assert isinstance(parsed, np.ndarray)
    assert np.allclose(data, parsed)


def test_binary_scalar_list_with_paren_byte() -> None:
    """Test binary data where ) byte (0x29) appears within the float data."""
    # Construct data where the int32 size boundary coincidentally lands on a ) byte
    # This tests the fix for int32 vs float64 ambiguity
    rng = np.random.default_rng(992)
    n = int(rng.integers(1, 200))
    data = rng.random(n)

    content = dumps({"FoamFile": {"format": "binary"}, None: data})
    result = parse_located(content)
    parsed = result[()].data

    assert isinstance(parsed, np.ndarray)
    assert np.allclose(data, parsed)


def test_binary_scalar_list_with_token_start_byte() -> None:
    """Test binary data where the first byte is an ASCII letter (token start char)."""
    # seed=539, n=52 produces binary data starting with 'H(' which used to trigger
    # FoamFileDecodeError via _parse_token depth tracking
    rng = np.random.default_rng(539)
    n = int(rng.integers(1, 200))
    data = rng.random(n)

    content = dumps({"FoamFile": {"format": "binary"}, None: data})
    result = parse_located(content)
    parsed = result[()].data

    assert isinstance(parsed, np.ndarray)
    assert np.allclose(data, parsed)


def test_binary_scalar_list_large_random() -> None:
    """Test many random binary float64 arrays for round-trip correctness."""
    for seed in range(500):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 200))
        data = rng.random(n)

        content = dumps({"FoamFile": {"format": "binary"}, None: data})
        result = parse_located(content)
        parsed = result[()].data

        assert isinstance(parsed, np.ndarray), f"seed={seed}: expected ndarray, got {type(parsed)}"
        assert np.allclose(data, parsed), f"seed={seed}: data mismatch"
