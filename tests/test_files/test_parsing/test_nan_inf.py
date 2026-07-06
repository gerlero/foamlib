import math

import numpy as np
from foamlib._files._parsing import ParsedFile


def test_parse_nan() -> None:
    """Test that NaN is parsed correctly in various formats."""
    # Standalone nan
    nan = ParsedFile(b"nan")[()]
    assert isinstance(nan, float)
    assert math.isnan(nan)
    nan = ParsedFile(b"NaN")[()]
    assert isinstance(nan, float)
    assert math.isnan(nan)
    nan = ParsedFile(b"NAN")[()]
    assert isinstance(nan, float)
    assert math.isnan(nan)

    # Signed nan
    nan = ParsedFile(b"+nan")[()]
    assert isinstance(nan, float)
    assert math.isnan(nan)
    nan = ParsedFile(b"-nan")[()]
    assert isinstance(nan, float)
    assert math.isnan(nan)

    # Uniform field with nan
    field = ParsedFile(b"uniform nan")[()]
    assert np.isnan(field)

    # Uniform tensor with nan
    tensor = ParsedFile(b"uniform (1 nan 3)")[()]
    assert isinstance(tensor, np.ndarray)
    assert tensor[0] == 1.0  # ty: ignore[invalid-argument-type]
    assert math.isnan(tensor[1])  # ty: ignore[invalid-argument-type]
    assert tensor[2] == 3.0  # ty: ignore[invalid-argument-type]

    # Tensor with multiple nan values
    tensor = ParsedFile(b"uniform (nan nan nan)")[()]
    assert isinstance(tensor, np.ndarray)
    assert all(math.isnan(x) for x in tensor)  # ty: ignore[invalid-argument-type, not-iterable]


def test_parse_infinity() -> None:
    """Test that infinity is parsed correctly in various formats."""
    # Standalone inf
    assert ParsedFile(b"inf")[()] == math.inf
    assert ParsedFile(b"Inf")[()] == math.inf
    assert ParsedFile(b"INF")[()] == math.inf

    # Standalone infinity
    assert ParsedFile(b"infinity")[()] == math.inf
    assert ParsedFile(b"Infinity")[()] == math.inf
    assert ParsedFile(b"INFINITY")[()] == math.inf

    # Signed inf
    assert ParsedFile(b"+inf")[()] == math.inf
    assert ParsedFile(b"-inf")[()] == -math.inf
    assert ParsedFile(b"+infinity")[()] == math.inf
    assert ParsedFile(b"-infinity")[()] == -math.inf

    # Uniform field with inf
    value = ParsedFile(b"uniform inf")[()]
    assert value == math.inf

    value = ParsedFile(b"uniform -inf")[()]
    assert value == -math.inf

    # Uniform tensor with inf
    tensor = ParsedFile(b"uniform (1 inf 3)")[()]
    assert isinstance(tensor, np.ndarray)
    assert tensor[0] == 1.0  # ty: ignore[invalid-argument-type]
    assert tensor[1] == math.inf  # ty: ignore[invalid-argument-type]
    assert tensor[2] == 3.0  # ty: ignore[invalid-argument-type]

    tensor = ParsedFile(b"uniform (-inf 0 inf)")[()]
    assert isinstance(tensor, np.ndarray)
    assert tensor[0] == -math.inf  # ty: ignore[invalid-argument-type]
    assert tensor[1] == 0.0  # ty: ignore[invalid-argument-type]
    assert tensor[2] == math.inf  # ty: ignore[invalid-argument-type]


def test_parse_nan_inf_in_lists() -> None:
    """Test that NaN and infinity are parsed correctly in lists."""
    # Note: Lists use numpy's fromstring which already supports nan/inf
    # These should already work, but let's verify

    # Float list with nan
    field = ParsedFile(b"nonuniform List<scalar> 3(1.0 nan 3.0)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field[0] == 1.0  # ty: ignore[invalid-argument-type]
    assert math.isnan(field[1])  # ty: ignore[invalid-argument-type]
    assert field[2] == 3.0  # ty: ignore[invalid-argument-type]

    # Float list with inf
    field = ParsedFile(b"nonuniform List<scalar> 3(inf -inf 0)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field[0] == math.inf  # ty: ignore[invalid-argument-type]
    assert field[1] == -math.inf  # ty: ignore[invalid-argument-type]
    assert field[2] == 0.0  # ty: ignore[invalid-argument-type]

    # Vector list with nan and inf
    field = ParsedFile(b"nonuniform List<vector> 2((1 nan 3) (inf -inf 0))")[()]
    assert isinstance(field, np.ndarray)
    assert field[0, 0] == 1.0  # ty: ignore[invalid-argument-type]
    assert math.isnan(field[0, 1])  # ty: ignore[invalid-argument-type]
    assert field[0, 2] == 3.0  # ty: ignore[invalid-argument-type]
    assert field[1, 0] == math.inf  # ty: ignore[invalid-argument-type]
    assert field[1, 1] == -math.inf  # ty: ignore[invalid-argument-type]
    assert field[1, 2] == 0.0  # ty: ignore[invalid-argument-type]


def test_parse_tokens_containing_nan_inf() -> None:
    """Test that tokens containing 'nan' or 'inf' as substrings are parsed as tokens."""
    # These should be parsed as tokens, not as NaN/infinity
    assert ParsedFile(b"inference")[()] == "inference"
    assert ParsedFile(b"infinity2")[()] == "infinity2"
    assert ParsedFile(b"nanometer")[()] == "nanometer"
    assert ParsedFile(b"info")[()] == "info"
