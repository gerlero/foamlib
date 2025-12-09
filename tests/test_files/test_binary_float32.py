"""Test binary float32 handling for standalone data and fields."""

import numpy as np
import pytest
from foamlib._files._serialization import dumps, normalized


def test_standalone_float32_normalization() -> None:
    """Test that standalone 1D float32 data can be normalized in binary format."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = normalized(data, keywords=(), format_="binary")
    assert result.dtype == np.float32
    assert np.array_equal(result, data)


def test_standalone_float64_normalization() -> None:
    """Test that standalone 1D float64 data can be normalized in binary format."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = normalized(data, keywords=(), format_="binary")
    assert result.dtype == np.float64
    assert np.array_equal(result, data)


def test_standalone_3d_float32_normalization() -> None:
    """Test that standalone 3D float32 data can be normalized in binary format."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = normalized(data, keywords=(), format_="binary")
    assert result.dtype == np.float32
    assert np.array_equal(result, data)


def test_standalone_3d_float64_normalization() -> None:
    """Test that standalone 3D float64 data can be normalized in binary format."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = normalized(data, keywords=(), format_="binary")
    assert result.dtype == np.float64
    assert np.array_equal(result, data)


def test_field_float32_normalization() -> None:
    """Test that field float32 data can be normalized in binary format."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    result = normalized(data, keywords=("internalField",), format_="binary")
    assert result.dtype == np.float32
    assert np.array_equal(result, data)


def test_field_float64_normalization() -> None:
    """Test that field float64 data can be normalized in binary format."""
    data = np.array([1.0, 2.0], dtype=np.float64)
    result = normalized(data, keywords=("internalField",), format_="binary")
    assert result.dtype == np.float64
    assert np.array_equal(result, data)


def test_standalone_float32_dumps() -> None:
    """Test that standalone 1D float32 data can be dumped in binary format."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    norm = normalized(data, keywords=(), format_="binary")
    dumped = dumps(norm, keywords=(), format_="binary")
    # Should produce 2 elements * 4 bytes = 8 bytes of data, plus count and parens
    assert isinstance(dumped, bytes)
    assert b"2(" in dumped
    # Verify the byte representation is float32
    assert len(dumped) == 11  # "2(" + 8 bytes + ")"


def test_standalone_float64_dumps() -> None:
    """Test that standalone 1D float64 data can be dumped in binary format."""
    data = np.array([1.0, 2.0], dtype=np.float64)
    norm = normalized(data, keywords=(), format_="binary")
    dumped = dumps(norm, keywords=(), format_="binary")
    # Should produce 2 elements * 8 bytes = 16 bytes of data, plus count and parens
    assert isinstance(dumped, bytes)
    assert b"2(" in dumped
    # Verify the byte representation is float64
    assert len(dumped) == 19  # "2(" + 16 bytes + ")"


def test_field_float32_dumps() -> None:
    """Test that field float32 data can be dumped in binary format."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    norm = normalized(data, keywords=("internalField",), format_="binary")
    dumped = dumps(norm, keywords=("internalField",), format_="binary")
    assert b"nonuniform List<scalar>" in dumped
    assert isinstance(dumped, bytes)
