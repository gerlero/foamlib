import math

import numpy as np
from foamlib._files._parsing import ParsedFile


def test_parse_nan() -> None:
    """Test that NaN is parsed correctly in various formats."""
    # Standalone nan
    assert math.isnan(ParsedFile(b"nan")[()])
    assert math.isnan(ParsedFile(b"NaN")[()])
    assert math.isnan(ParsedFile(b"NAN")[()])

    # Signed nan
    assert math.isnan(ParsedFile(b"+nan")[()])
    assert math.isnan(ParsedFile(b"-nan")[()])

    # Uniform field with nan
    value = ParsedFile(b"uniform nan")[()]
    assert math.isnan(value)

    # Uniform tensor with nan
    tensor = ParsedFile(b"uniform (1 nan 3)")[()]
    assert isinstance(tensor, np.ndarray)
    assert tensor[0] == 1.0
    assert math.isnan(tensor[1])
    assert tensor[2] == 3.0

    # Tensor with multiple nan values
    tensor = ParsedFile(b"uniform (nan nan nan)")[()]
    assert isinstance(tensor, np.ndarray)
    assert all(math.isnan(x) for x in tensor)


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
    assert tensor[0] == 1.0
    assert tensor[1] == math.inf
    assert tensor[2] == 3.0

    tensor = ParsedFile(b"uniform (-inf 0 inf)")[()]
    assert isinstance(tensor, np.ndarray)
    assert tensor[0] == -math.inf
    assert tensor[1] == 0.0
    assert tensor[2] == math.inf


def test_parse_nan_inf_in_lists() -> None:
    """Test that NaN and infinity are parsed correctly in lists."""
    # Note: Lists use numpy's fromstring which already supports nan/inf
    # These should already work, but let's verify

    # Float list with nan
    field = ParsedFile(b"nonuniform List<scalar> 3(1.0 nan 3.0)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field[0] == 1.0
    assert math.isnan(field[1])
    assert field[2] == 3.0

    # Float list with inf
    field = ParsedFile(b"nonuniform List<scalar> 3(inf -inf 0)")[()]
    assert isinstance(field, np.ndarray)
    assert field.dtype == float
    assert field[0] == math.inf
    assert field[1] == -math.inf
    assert field[2] == 0.0

    # Vector list with nan and inf
    field = ParsedFile(b"nonuniform List<vector> 2((1 nan 3) (inf -inf 0))")[()]
    assert isinstance(field, np.ndarray)
    assert field[0, 0] == 1.0
    assert math.isnan(field[0, 1])
    assert field[0, 2] == 3.0
    assert field[1, 0] == math.inf
    assert field[1, 1] == -math.inf
    assert field[1, 2] == 0.0


def test_parse_tokens_containing_nan_inf() -> None:
    """Test that tokens containing 'nan' or 'inf' as substrings are parsed as tokens."""
    # These should be parsed as tokens, not as NaN/infinity
    assert ParsedFile(b"inference")[()] == "inference"
    assert ParsedFile(b"infinity2")[()] == "infinity2"
    assert ParsedFile(b"nanometer")[()] == "nanometer"
    assert ParsedFile(b"info")[()] == "info"
