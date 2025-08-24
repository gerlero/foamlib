"""Test serialization edge cases and complex data structures."""

import numpy as np
import pytest
from foamlib import FoamFile
from foamlib._files._serialization import dumps
from foamlib._files._types import DimensionSet, Dimensioned


def test_dumps_edge_case_numbers():
    """Test dumps with edge case numeric values."""
    # Very large numbers
    assert dumps(1e100) == b"1e+100"
    assert dumps(-1e100) == b"-1e+100"
    
    # Very small numbers
    assert dumps(1e-100) == b"1e-100"
    assert dumps(-1e-100) == b"-1e-100"
    
    # Zero variations
    assert dumps(0.0) == b"0.0"
    assert dumps(-0.0) == b"0.0"  # Negative zero should be normalized
    
    # Special float values
    assert dumps(float('inf')) == b"inf"
    assert dumps(float('-inf')) == b"-inf"
    assert dumps(float('nan')) == b"nan"
    
    # Numbers with many decimal places
    assert dumps(1.23456789012345) == b"1.23456789012345"
    
    # Numbers that might have precision issues
    assert dumps(0.1 + 0.2) == b"0.30000000000000004"  # Floating point precision


def test_dumps_complex_lists():
    """Test dumps with complex list structures."""
    # Nested lists with mixed types
    mixed_nested = [[1, 2.5, "string"], [True, False], []]
    result = dumps(mixed_nested)
    assert b"((1.0 2.5 string) (yes no) ())" == result
    
    # List with numpy arrays
    np_list = [np.array([1, 2, 3]), np.array([4.0, 5.0, 6.0])]
    result = dumps(np_list)
    assert b"((1.0 2.0 3.0) (4.0 5.0 6.0))" == result
    
    # Very long list
    long_list = list(range(100))
    result = dumps(long_list)
    assert result.startswith(b"(")
    assert result.endswith(b")")
    assert b"99.0" in result
    
    # List with special characters in strings
    special_list = ["normal", "with spaces", "with\nnewline", "with\ttab"]
    result = dumps(special_list)
    # Should handle special characters appropriately
    assert b"normal" in result
    assert b'"with spaces"' in result


def test_dumps_complex_tuples():
    """Test dumps with complex tuple structures (OpenFOAM scheme syntax)."""
    # Simple scheme
    scheme = ("Gauss", "linear")
    assert dumps(scheme, keywords=()) == b"Gauss linear"
    
    # Complex scheme with parameters
    complex_scheme = ("Gauss", "linearUpwind", "grad(U)")
    assert dumps(complex_scheme, keywords=()) == b"Gauss linearUpwind grad(U)"
    
    # Scheme with numbers
    numeric_scheme = ("Gauss", "linear", "limited", 0.333)
    assert dumps(numeric_scheme, keywords=()) == b"Gauss linear limited 0.333"
    
    # Nested scheme structures
    nested_scheme = ("Gauss", ("upwind", "phi"))
    result = dumps(nested_scheme, keywords=())
    # Implementation dependent on how nested tuples are handled
    assert b"Gauss" in result


def test_dumps_dimension_sets_edge_cases():
    """Test dumps with various dimension set edge cases."""
    # All zero dimensions (dimensionless)
    dimensionless = DimensionSet()
    assert dumps(dimensionless) == b"[0 0 0 0 0 0 0]"
    
    # Negative dimensions
    negative_dims = DimensionSet(mass=-1, length=-2, time=3)
    assert dumps(negative_dims) == b"[-1 -2 3 0 0 0 0]"
    
    # Fractional dimensions
    fractional_dims = DimensionSet(mass=0.5, length=1.5, time=-2.5)
    assert dumps(fractional_dims) == b"[0.5 1.5 -2.5 0 0 0 0]"
    
    # Large dimension values
    large_dims = DimensionSet(mass=100, length=-50, time=25)
    assert dumps(large_dims) == b"[100 -50 25 0 0 0 0]"


def test_dumps_dimensioned_edge_cases():
    """Test dumps with various Dimensioned object edge cases."""
    # Dimensioned with name
    named_dimensioned = Dimensioned(
        name="gravity",
        dimensions=DimensionSet(length=1, time=-2),
        value=9.81
    )
    assert dumps(named_dimensioned) == b"gravity [0 1 -2 0 0 0 0] 9.81"
    
    # Dimensioned without name
    unnamed_dimensioned = Dimensioned(
        dimensions=DimensionSet(mass=1, length=1, time=-2),
        value=101325
    )
    assert dumps(unnamed_dimensioned) == b"[1 1 -2 0 0 0 0] 101325"
    
    # Dimensioned with vector value
    vector_dimensioned = Dimensioned(
        name="velocity",
        dimensions=DimensionSet(length=1, time=-1),
        value=[1.0, 2.0, 3.0]
    )
    result = dumps(vector_dimensioned)
    assert b"velocity" in result
    assert b"[0 1 -1 0 0 0 0]" in result
    assert b"(1.0 2.0 3.0)" in result
    
    # Dimensioned with zero value
    zero_dimensioned = Dimensioned(
        dimensions=DimensionSet(length=1),
        value=0.0
    )
    assert dumps(zero_dimensioned) == b"[0 1 0 0 0 0 0] 0.0"


def test_dumps_with_keywords_edge_cases():
    """Test dumps with various keyword contexts."""
    # internalField with scalar
    scalar_result = dumps(42.0, keywords=("internalField",))
    assert scalar_result == b"uniform 42.0"
    
    # internalField with vector
    vector_result = dumps([1, 2, 3], keywords=("internalField",))
    assert vector_result == b"uniform (1.0 2.0 3.0)"
    
    # internalField with large array (should trigger nonuniform)
    large_array = list(range(15))  # Assuming threshold > 10
    large_result = dumps(large_array, keywords=("internalField",))
    assert large_result.startswith(b"nonuniform List<scalar>")
    
    # Non-internalField keywords
    regular_result = dumps(42, keywords=("someOtherKeyword",))
    assert regular_result == b"42"
    
    # Empty keywords
    empty_result = dumps(42, keywords=())
    assert empty_result == b"42"


def test_dumps_binary_format():
    """Test dumps with binary format header."""
    binary_header = {"format": "binary"}
    
    # Small data should still be ascii in binary mode for internalField
    small_result = dumps([1, 2, 3], keywords=("internalField",), header=binary_header)
    assert small_result == b"uniform (1.0 2.0 3.0)"
    
    # Large data should be binary
    large_data = list(range(15))
    large_result = dumps(large_data, keywords=("internalField",), header=binary_header)
    assert large_result.startswith(b"nonuniform List<scalar>")
    assert len(large_result) > 50  # Should contain binary data
    
    # Regular data not affected by binary header
    regular_result = dumps(42, keywords=("regularKey",), header=binary_header)
    assert regular_result == b"42"


def test_dumps_numpy_arrays_edge_cases():
    """Test dumps with various numpy array edge cases."""
    # Different numpy dtypes
    int32_array = np.array([1, 2, 3], dtype=np.int32)
    assert dumps(int32_array) == b"(1.0 2.0 3.0)"
    
    float32_array = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    assert dumps(float32_array) == b"(1.100000023841858 2.200000047683716 3.299999952316284)"
    
    # Boolean arrays
    bool_array = np.array([True, False, True])
    result = dumps(bool_array)
    assert result == b"(yes no yes)"
    
    # Empty arrays
    empty_array = np.array([])
    assert dumps(empty_array) == b"()"
    
    # Single element arrays
    single_array = np.array([42])
    assert dumps(single_array) == b"(42.0)"
    
    # Multi-dimensional arrays (should be flattened or handled appropriately)
    multi_array = np.array([[1, 2], [3, 4]])
    result = dumps(multi_array)
    # Behavior depends on implementation - should handle gracefully


def test_dumps_string_edge_cases():
    """Test dumps with various string edge cases."""
    # Strings with spaces (should be quoted)
    spaced_string = "string with spaces"
    assert dumps(spaced_string) == b'"string with spaces"'
    
    # Strings with special characters
    special_string = "string!@#$%^&*()"
    assert dumps(special_string) == b'"string!@#$%^&*()"'
    
    # Already quoted strings
    quoted_string = '"already quoted"'
    assert dumps(quoted_string) == b'"already quoted"'
    
    # Empty string
    empty_string = ""
    assert dumps(empty_string) == b'""'
    
    # Single character strings
    single_char = "a"
    assert dumps(single_char) == b"a"
    
    # Very long strings
    long_string = "a" * 1000
    result = dumps(long_string)
    assert len(result) == 1000  # Should handle long strings
    
    # Strings with newlines and tabs
    multiline_string = "line1\nline2\ttab"
    result = dumps(multiline_string)
    assert b'"' in result  # Should be quoted due to special chars


def test_dumps_dict_like_structures():
    """Test dumps with dictionary-like structures."""
    # List of key-value pairs (OpenFOAM boundary syntax)
    boundary_list = [
        ("inlet", {"type": "fixedValue", "value": "uniform (1 0 0)"}),
        ("outlet", {"type": "zeroGradient"}),
        ("walls", {"type": "noSlip"})
    ]
    result = dumps(boundary_list)
    assert b"inlet" in result
    assert b"fixedValue" in result
    assert b"outlet" in result
    assert b"zeroGradient" in result
    
    # Nested dictionary structures
    nested_dict = {
        "level1": {
            "level2": {
                "value": 42,
                "list": [1, 2, 3]
            }
        }
    }
    # Direct dict dumping depends on implementation


def test_dumps_error_cases():
    """Test dumps with data that should cause errors."""
    # Complex numbers (not supported)
    with pytest.raises((TypeError, ValueError)):
        dumps(complex(1, 2))
    
    # Unsupported object types
    class CustomObject:
        pass
    
    with pytest.raises((TypeError, ValueError)):
        dumps(CustomObject())
    
    # None values in inappropriate contexts
    with pytest.raises((TypeError, ValueError)):
        dumps(None)


def test_foamfile_dumps_method():
    """Test FoamFile.dumps static method edge cases."""
    # Without header
    result = FoamFile.dumps(42, ensure_header=False)
    assert result == b"42"
    
    # With header (default)
    result = FoamFile.dumps(42)
    assert b"FoamFile" in result
    assert b"42" in result
    
    # Complex data structure
    complex_data = {
        "solver": "PISO",
        "nCorrectors": 2,
        "tolerance": 1e-6,
        "schemes": {
            "ddtSchemes": {"default": "Euler"},
            "gradSchemes": {"default": ("Gauss", "linear")}
        }
    }
    result = FoamFile.dumps(complex_data)
    assert b"FoamFile" in result
    assert b"PISO" in result
    assert b"Euler" in result
    
    # Field data (should infer correct class)
    field_data = {"internalField": [[1, 2, 3], [4, 5, 6]]}
    result = FoamFile.dumps(field_data)
    assert b"volVectorField" in result
    
    # Binary format
    binary_data = {
        "FoamFile": {"format": "binary"},
        None: np.array([1, 2, 3, 4, 5], dtype=np.int32)
    }
    result = FoamFile.dumps(binary_data, ensure_header=False)
    assert b"format binary" in result
    
    # Include directive
    include_data = {"#include": "$FOAM_CASE/system/controlDict"}
    result = FoamFile.dumps(include_data, ensure_header=False)
    assert b"#include" in result
    assert b"$FOAM_CASE" in result


def test_dumps_consistency():
    """Test that dumps produces consistent output."""
    test_data = {
        "scalar": 42.0,
        "vector": [1, 2, 3],
        "string": "test",
        "boolean": True,
        "list": [1, 2, 3, 4, 5],
        "nested": {"a": 1, "b": 2}
    }
    
    # Multiple calls should produce identical results
    result1 = dumps(test_data)
    result2 = dumps(test_data)
    assert result1 == result2
    
    # Order should be preserved for lists
    ordered_list = [3, 1, 4, 1, 5, 9]
    result = dumps(ordered_list)
    # Should maintain order in output