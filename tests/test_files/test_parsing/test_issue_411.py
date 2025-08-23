"""Test for issue #411: Parser fails with | operator in parentheses."""

from foamlib._files._parsing import Parsed


def test_issue_411_pipe_operator() -> None:
    """Test that expressions with | operator from issue #411 can now be parsed.

    These test cases were originally failing with parsing errors due to the
    pipe character '|' not being properly handled within parentheses.
    """

    # Test case 1: The exact example from issue #411
    parsed = Parsed(b"""
        laplacianSchemes
        {
            laplacian((1|A(U)),p) Gauss linear limited 0.333;
        }
    """)
    assert parsed[("laplacianSchemes", "laplacian((1|A(U)),p)")] == (
        "Gauss",
        "linear",
        "limited",
        0.333,
    )

    # Test case 2: More complex expression with pipe in div schemes
    parsed = Parsed(b"""
        divSchemes
        {
            div((rho*thermo:mu|rho),U) Gauss linear;
        }
    """)
    assert parsed[("divSchemes", "div((rho*thermo:mu|rho),U)")] == ("Gauss", "linear")

    # Test case 3: Pipe in laplacian schemes
    parsed = Parsed(b"""
        laplacianSchemes
        {
            laplacian((DT|rho),T) Gauss linear corrected;
        }
    """)
    assert parsed[("laplacianSchemes", "laplacian((DT|rho),T)")] == (
        "Gauss",
        "linear",
        "corrected",
    )

    # Test case 4: Simple pipe in function call
    parsed = Parsed(b"""
        interpolationSchemes
        {
            interpolate(rho|U) linear;
        }
    """)
    assert parsed[("interpolationSchemes", "interpolate(rho|U)")] == "linear"

    # Test case 5: Multiple pipes and complex nesting
    parsed = Parsed(b"""
        divSchemes
        {
            div(((rho*(thermo:mu|rho))*dev2(T(grad(U))))) Gauss linear;
        }
    """)
    assert parsed[("divSchemes", "div(((rho*(thermo:mu|rho))*dev2(T(grad(U)))))")] == (
        "Gauss",
        "linear",
    )

    # Test case 6: Pipe with colon operator (from issue #393 tests)
    parsed = Parsed(b"""
        divSchemes
        {
            div(((rho*(thermo:mu|rho))*dev2(T(grad(U))))) Gauss linear;
        }
    """)
    assert parsed[("divSchemes", "div(((rho*(thermo:mu|rho))*dev2(T(grad(U)))))")] == (
        "Gauss",
        "linear",
    )


def test_issue_411_edge_cases() -> None:
    """Test edge cases related to pipe operator."""

    # Test pipe at start of parentheses
    parsed = Parsed(b"""
        schemes
        {
            test((|value),param) result;
        }
    """)
    assert parsed[("schemes", "test((|value),param)")] == "result"

    # Test pipe at end of parentheses
    parsed = Parsed(b"""
        schemes
        {
            test((value|),param) result;
        }
    """)
    assert parsed[("schemes", "test((value|),param)")] == "result"

    # Test multiple consecutive pipes
    parsed = Parsed(b"""
        schemes
        {
            test((a||b),param) result;
        }
    """)
    assert parsed[("schemes", "test((a||b),param)")] == "result"

    # Test pipe with other special characters
    parsed = Parsed(b"""
        schemes
        {
            test((a|b&c.d:e),param) result;
        }
    """)
    assert parsed[("schemes", "test((a|b&c.d:e),param)")] == "result"
