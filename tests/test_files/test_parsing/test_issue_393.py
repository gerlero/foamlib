"""Test for issue #393: Parsing issues with complex keys."""

from foamlib._files._parsing import Parsed


def test_issue_393_complex_keys() -> None:
    """Test that complex keys from issue #393 can now be parsed.

    These test cases were originally failing with parsing errors like:
    "Expected end of text, found '{' (at char 1498), (line:42, col:1)"
    """

    # Test case 1: dot in parentheses - was failing due to the period in .T()
    parsed = Parsed(b"""
        divSchemes
        {
            div((muEff*dev2(grad(U).T()))) Gauss linear;
        }
    """)
    assert parsed[("divSchemes", "div((muEff*dev2(grad(U).T())))")] == (
        "Gauss",
        "linear",
    )

    # Test case 2: double asterisk in parentheses - was failing due to complex nesting
    parsed = Parsed(b"""
        divSchemes
        {
            div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
        }
    """)
    assert parsed[("divSchemes", "div(((rho*nuEff)*dev2(T(grad(U)))))")] == (
        "Gauss",
        "linear",
    )

    # Test case 3: complex expression with colon and pipe - was failing due to special chars
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

    # Test case 4: comma in parentheses - was failing due to comma inside parentheses
    parsed = Parsed(b"""
        laplacianSchemes
        {
            laplacian((rho*DomegaEff),omega) omegaGauss linear limited 0.333;
        }
    """)
    assert parsed[("laplacianSchemes", "laplacian((rho*DomegaEff),omega)")] == (
        "omegaGauss",
        "linear",
        "limited",
        0.333,
    )

    # Test case 5: multiple comma cases from the original issue
    parsed = Parsed(b"""
        laplacianSchemes
        {
            laplacian((rho*DkEff),k) Gauss linear limited 0.333;
            laplacian((rho*DepsilonEff),omega) Gauss linear limited 0.333;
            laplacian((rho*DnuTildaEff),k) Gauss linear limited 0.333;
        }
    """)
    assert parsed[("laplacianSchemes", "laplacian((rho*DkEff),k)")] == (
        "Gauss",
        "linear",
        "limited",
        0.333,
    )
    assert parsed[("laplacianSchemes", "laplacian((rho*DepsilonEff),omega)")] == (
        "Gauss",
        "linear",
        "limited",
        0.333,
    )
    assert parsed[("laplacianSchemes", "laplacian((rho*DnuTildaEff),k)")] == (
        "Gauss",
        "linear",
        "limited",
        0.333,
    )


def test_original_working_cases_still_work() -> None:
    """Ensure that the fix doesn't break originally working cases."""
    parsed = Parsed(b"""
        divSchemes
        {
            div((rho*R)) Gauss linear;
            div(phi,omega) Gauss linearUpwind grad(omega);
            div(phi,U) Gauss linearUpwind grad(U);
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
        }

        laplacianSchemes
        {
            default         Gauss linear limited 0.333;
        }
    """)
    assert parsed[("divSchemes", "div((rho*R))")] == ("Gauss", "linear")
    assert parsed[("divSchemes", "div(phi,omega)")] == (
        "Gauss",
        "linearUpwind",
        "grad(omega)",
    )
    assert parsed[("divSchemes", "div(phi,U)")] == ("Gauss", "linearUpwind", "grad(U)")
    assert parsed[("divSchemes", "div((nuEff*dev2(T(grad(U)))))")] == (
        "Gauss",
        "linear",
    )
    assert parsed[("laplacianSchemes", "default")] == (
        "Gauss",
        "linear",
        "limited",
        0.333,
    )
