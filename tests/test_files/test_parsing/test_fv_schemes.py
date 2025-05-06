# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_fv_schemes.py

from pathlib import Path
from textwrap import dedent

from foamlib import FoamFile

contents = dedent(
    r"""
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        object      fvSchemes;
    }

    ddtSchemes
    {
        default    Euler;
    }

    gradSchemes
    {
    }

    divSchemes
    {
        div(phi,T)        Gauss limitedLinear 1;
        div(phi,U)        foo;
        div(phi.b,k.b)    Gauss limitedLinear 1;
        div(phi,k)        $turbulence;
    }

    laplacianSchemes
    {
    }

    interpolationSchemes
    {
    }

    snGradSchemes
    {
    }

    fluxRequired
    {
        default    no;
        p_rbgh;
    }
"""
)


def test_simple(tmp_path: Path) -> None:
    path = tmp_path / "fvSchemes"
    path.write_text(contents)

    fv_schemes = FoamFile(path)

    assert fv_schemes["ddtSchemes", "default"] == "Euler"
    assert not fv_schemes["gradSchemes"]
    assert fv_schemes["divSchemes", "div(phi,T)"] == ("Gauss", "limitedLinear", 1)
    assert fv_schemes["divSchemes", "div(phi,U)"] == "foo"
    assert fv_schemes["divSchemes", "div(phi.b,k.b)"] == ("Gauss", "limitedLinear", 1)
    assert fv_schemes["divSchemes", "div(phi,k)"] == "$turbulence"
    assert not fv_schemes["laplacianSchemes"]
    assert not fv_schemes["interpolationSchemes"]
    assert not fv_schemes["snGradSchemes"]
    assert fv_schemes["fluxRequired", "default"] is False
    assert fv_schemes["fluxRequired", "p_rbgh"] == ""
