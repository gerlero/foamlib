"""Test parsing of expressions containing pipe character (|) - Issue #411."""

from foamlib._files._parsing import Parsed


def test_pipe_in_laplacian_expression():
    """Test that pipe characters work in laplacian expressions like laplacian((1|A(U)),p)."""
    # This was the original failing case from issue #411
    parsed = Parsed(b"""
    laplacianSchemes
    {
        laplacian((1|A(U)),p) Gauss linear corrected;
        laplacian((nu|nuTilda),k) Gauss linear corrected;
    }
    """)
    
    assert parsed[("laplacianSchemes", "laplacian((1|A(U)),p)")] == ("Gauss", "linear", "corrected")
    assert parsed[("laplacianSchemes", "laplacian((nu|nuTilda),k)")] == ("Gauss", "linear", "corrected")


def test_pipe_in_complex_parentheses():
    """Test pipe characters in complex nested parenthesized expressions."""
    parsed = Parsed(b"""
    test
    {
        func((a|b),c) value1;
        func((x|y|z),d) value2;
        func(((nested|expr)),p) value3;
    }
    """)
    
    assert parsed[("test", "func((a|b),c)")] == "value1"
    assert parsed[("test", "func((x|y|z),d)")] == "value2"
    assert parsed[("test", "func(((nested|expr)),p)")] == "value3"


def test_pipe_in_function_objects():
    """Test pipe characters in function object expressions."""
    parsed = Parsed(b"""
    functions
    {
        gradient((1|A(U)),p)
        {
            type    grad;
            field   U;
        }
    }
    """)
    
    assert parsed[("functions", "gradient((1|A(U)),p)", "type")] == "grad"
    assert parsed[("functions", "gradient((1|A(U)),p)", "field")] == "U"


def test_backwards_compatibility():
    """Ensure existing pipe character usage still works."""
    # Test quoted identifiers (should continue to work)
    parsed = Parsed(b"""
    equations
    {
        "(U|e|k).*"  0.7;
        "(U|e|k|epsilon).*" table ((0 0.4) (0.5 0.7));
    }
    """)
    
    assert parsed[("equations", '"(U|e|k).*"')] == 0.7
    assert parsed[("equations", '"(U|e|k|epsilon).*"')][0] == "table"