# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_parser_advanced.py

import numpy as np
import pytest
from foamlib._files._parsing import Parsed


@pytest.mark.xfail(reason="Not currently supported")
def test_dict_strange_name() -> None:
    parsed = Parsed(
        b"""
        div(phi,ft_b_ha_hau) Gauss multivariateSelection
        {
            ft              limitedLinear01 1;
            b               limitedLinear01 1;
        }
    """
    )
    assert parsed[
        (("div(phi,ft_b_ha_hau)", "Gauss", "multivariateSelection"), "ft")  # type: ignore[index]
    ] == ("limitedLinear01", 1)
    assert parsed[
        (("div(phi,ft_b_ha_hau)", "Gauss", "multivariateSelection"), "b")  # type: ignore[index]
    ] == ("limitedLinear01", 1)


def test_dict_strange_keys() -> None:
    parsed = Parsed(
        b"""
        div(phi,U)      Gauss linear;
        divSchemes
        {
            field       cylindrical(U)Mean;
            default         none;
            div(phi,U)      Gauss linear;
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
            ".*"           1;
            div(rhoPhi,U)   Gauss cellCoBlended 2 linearUpwind grad(U) 5 upwind;
        }
    """
    )
    assert parsed[("divSchemes", "div(phi,U)")] == ("Gauss", "linear")
    assert parsed[("divSchemes", "div((nuEff*dev2(T(grad(U)))))")] == (
        "Gauss",
        "linear",
    )
    assert parsed[("divSchemes", '".*"')] == 1
    assert parsed[("divSchemes", "div(rhoPhi,U)")] == (
        "Gauss",
        "cellCoBlended",
        2,
        "linearUpwind",
        "grad(U)",
        5,
        "upwind",
    )


def test_var_value_with_space() -> None:
    parsed = Parsed(
        b"""
        laplacianSchemes
        {
            default         Gauss linear corrected;
        }
    """
    )
    assert parsed[("laplacianSchemes", "default")] == ("Gauss", "linear", "corrected")


@pytest.mark.xfail(reason="Not currently supported")
def test_strange_dict_macro() -> None:
    parsed = Parsed(
        b"""
        relaxationFactors { $relaxationFactors-SIMPLE }
        """
    )
    assert parsed[("relaxationFactors",)] == "$relaxationFactors-SIMPLE"


@pytest.mark.xfail(reason="Not currently supported")
def test_directive_eval() -> None:
    parsed = Parsed(
        b"""
        transform
        {
            origin  (#eval{0.5 * $SLAB_OFFSET} 0 0);
            rotation none;
        }
        """
    )
    assert parsed[("transform", "origin")] == ["#eval{0.5 * $SLAB_OFFSET}", 0, 0]
    assert parsed[("transform", "rotation")] == "none"


@pytest.mark.xfail(reason="Not currently supported")
def test_directive_if() -> None:
    Parsed(
        b"""
        #if 0
        xin     #eval{ $xin / 5 };
        xout    #eval{ $xout / 5 };
        zmax    #eval{ $zmax / 5

        nxin    #eval{ round ($nxin / 5) };
        nxout   #eval{ round ($nxout / 5) };
        nz      #eval{ round ($nz / 5) };
        #endif
        """
    )


@pytest.mark.xfail(reason="Not currently supported")
def test_directive_if_in_file() -> None:
    Parsed(
        b"""
        #if 0
        xin     #eval{ $xin / 5 };
        xout    #eval{ $xout / 5 };
        zmax    #eval{ $zmax / 5

        nxin    #eval{ round ($nxin / 5) };
        nxout   #eval{ round ($nxout / 5) };
        nz      #eval{ round ($nz / 5) };
        #endif

        zmin    #eval{ -$zmax };
        """
    )


@pytest.mark.xfail(reason="Not currently supported")
def test_macro_with_dict() -> None:
    parsed = Parsed(
        b"""
        rInner45    ${{ $rInner * sqrt(0.5) }};
        rOuter45    ${{ $rOuter * sqrt(0.5) }};
        xmin        ${{ -$xmax }};
        """
    )
    assert parsed[("rInner45",)] == "${{ $rInner * sqrt(0.5) }}"
    assert parsed[("rOuter45",)] == "${{ $rOuter * sqrt(0.5) }}"
    assert parsed[("xmin",)] == "${{ -$xmax }}"


def test_directive_strange() -> None:
    parsed = Parsed(
        b"""
        #remove ( "r(Inner|Outer).*"  "[xy](min|max)" )
        """
    )
    assert parsed[("#remove",)] == ['"r(Inner|Outer).*"', '"[xy](min|max)"']


@pytest.mark.xfail(reason="Not currently supported")
def test_directive_with_macro() -> None:
    parsed = Parsed(
        b"""
        timeStart       #eval{ 0.1 * ${/endTime} };
        """
    )
    assert parsed[("timeStart",)] == "#eval{ 0.1 * ${/endTime} }"


def test_strange_assignment() -> None:
    parsed = Parsed(
        b"""
        divSchemes
        {
            div(phi,U)      Gauss DEShybrid
                linear                    // scheme 1
                linearUpwind grad(U)      // scheme 2
                hmax
                0.65                      // DES coefficient, typically = 0.65
                1                         // Reference velocity scale
                0.028                     // Reference length scale
                0                         // Minimum sigma limit (0-1)
                1                         // Maximum sigma limit (0-1)
                1; // 1.0e-03;                  // Limiter of B function, typically 1e-03
        }
        """
    )
    assert parsed[("divSchemes", "div(phi,U)")] == (
        "Gauss",
        "DEShybrid",
        "linear",
        "linearUpwind",
        "grad(U)",
        "hmax",
        0.65,
        1,
        0.028,
        0,
        1,
        1,
    )


def test_unnamed_dict_in_list() -> None:
    parsed = Parsed(
        b"""
        drag
        (
            (air water)
            {
                type                    blended;
                residualPhaseFraction   0.001;
                residualSlip            0.001;
            }
        );
        """
    )
    assert parsed[("drag",)] == [
        (
            ["air", "water"],
            {"type": "blended", "residualPhaseFraction": 0.001, "residualSlip": 0.001},
        )
    ]


def test_unnamed_dict_in_list1() -> None:
    parsed = Parsed(
        b"""
        features
        (
            {
                file     "geom.extendedFeatureEdgeMesh";
                level    1;
            }
        );
        """
    )
    assert parsed[("features",)] == [
        {"file": '"geom.extendedFeatureEdgeMesh"', "level": 1}
    ]


@pytest.mark.xfail(reason="Not currently supported")
def test_list_name_eq() -> None:
    Parsed(
        b"""
        value #eval
        {
            -9.81 * vector
            (
                sin(degToRad($alphax)),
                sin(degToRad($alpha)),
                cos(degToRad($alpha))
            )
        };
        """
    )


def test_list_triple_named() -> None:
    parsed = Parsed(
        b"""
        velocity-inlet-5
        {
            type            fixedValue;
            value           uniform (1 0 0);
        }
        """
    )
    assert parsed[("velocity-inlet-5", "type")] == "fixedValue"
    assert parsed[("velocity-inlet-5", "value")] == pytest.approx([1, 0, 0])


def test_assignment_strange_name() -> None:
    parsed = Parsed(
        b"""
        equations
        {
            "(U|e|k).*"  0.7;
            "(U|e|k|epsilon).*" table ((0 0.4) (0.5 0.7));
        }
    """
    )
    assert parsed[("equations", '"(U|e|k).*"')] == 0.7
    assert isinstance(parsed[("equations", '"(U|e|k|epsilon).*"')], tuple)
    assert parsed[("equations", '"(U|e|k|epsilon).*"')][0] == "table"
    assert np.array_equal(
        parsed[("equations", '"(U|e|k|epsilon).*"')][1],  # type: ignore[arg-type]
        [[0, 0.4], [0.5, 0.7]],
    )


@pytest.mark.xfail(reason="Not currently supported")
def test_code_with_directive_and_macro() -> None:
    parsed = Parsed(
        b"""
        timeStart  #eval #{ 1.0/3.0 * ${/endTime} #};
        U
        {
            mean          on;
            prime2Mean    on;
            base          time;
        }
        """
    )
    assert parsed[("timeStart",)] == "#eval #{ 1.0/3.0 * ${/endTime} #}"
    assert parsed[("U", "mean")] is True
    assert parsed[("U", "prime2Mean")] is True
    assert parsed[("U", "base")] == "time"


@pytest.mark.xfail(reason="Not currently supported")
def test_code_with_directive() -> None:
    parsed = Parsed(
        b"""
        nx  #eval #{ round(5 * $NSLABS) #};
        """
    )
    assert parsed[("nx",)] == "#eval #{ round(5 * $NSLABS) #}"


def test_list_u() -> None:
    parsed = Parsed(
        b"""
        FoamFile
        {
            version     2.0;
        }
        (
        (4.507730000e+00 1.799630000e+00 0.000000000e+00)
        (6.062080000e+00 2.408310000e+00 0.000000000e+00)
        (6.874000000e+00 2.720790000e+00 0.000000000e+00)
        (7.429290000e+00 2.931000000e+00 0.000000000e+00)
        (7.850950000e+00 3.088050000e+00 0.000000000e+00)
        (8.192020000e+00 3.213060000e+00 0.000000000e+00)
        (1.750000000e+01 1.925590000e-09 0.000000000e+00)
        (1.750000000e+01 6.810450000e-12 0.000000000e+00)
        (1.750000000e+01 6.810450000e-12 0.000000000e+00)
        )
        """
    )
    data = parsed[()]
    assert isinstance(data, np.ndarray)
    assert data.shape == (9, 3)
    assert np.array_equal(
        data,
        [
            [4.507730000e00, 1.799630000e00, 0.000000000e00],
            [6.062080000e00, 2.408310000e00, 0.000000000e00],
            [6.874000000e00, 2.720790000e00, 0.000000000e00],
            [7.429290000e00, 2.931000000e00, 0.000000000e00],
            [7.850950000e00, 3.088050000e00, 0.000000000e00],
            [8.192020000e00, 3.213060000e00, 0.000000000e00],
            [1.750000000e01, 1.925590000e-09, 0.000000000e00],
            [1.750000000e01, 6.810450000e-12, 0.000000000e00],
            [1.750000000e01, 6.810450000e-12, 0.000000000e00],
        ],
    )


def test_list_as_write_cell_centers() -> None:
    parsed = Parsed(
        b"""
        value           nonuniform List<scalar>
        2
        (
            47.619
            142.857
        );
        """
    )
    assert parsed[("value",)] == pytest.approx([47.619, 142.857])


def test_list_as_write_cell_centers_short() -> None:
    parsed = Parsed(b"value           nonuniform List<scalar> 4(250 750 1250 1750);")
    assert parsed[("value",)] == pytest.approx([250, 750, 1250, 1750])
