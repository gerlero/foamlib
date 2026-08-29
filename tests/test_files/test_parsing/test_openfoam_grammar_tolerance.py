"""OpenFOAM-grammar tolerance parity tests.

foamlib's FoamFile parser was stricter than OpenFOAM's token-based parser at
statement boundaries, rejecting a family of legal dictionary forms (100 of the
4234 dictionary files shipped with OpenFOAM-11). Each case mirrors the
OpenFOAM-11 parser source (entryIO.C, primitiveEntryIO.C, functionEntry.C,
dictionaryIO.C, dimensionSetIO.C) and round-trips through the data model.
"""

import pytest

from foamlib import Dimensioned, DimensionSet, FoamFile, FoamFileDecodeError
from foamlib._files._parsing import ParsedFile, parse
from foamlib.typing import FileDict


def _file_dict(contents: bytes) -> FileDict:
    return parse(contents, target=FileDict)  # ty: ignore[invalid-argument-type]


def test_spurious_semicolon_after_subdictionary() -> None:
    # entry::getKeyword: "Read the next valid token discarding spurious ';'s"
    assert _file_dict(b"a {x 1;}; b 2;") == {"a": {"x": 1}, "b": 2}
    assert _file_dict(b"outer {a {x 1;}; b 2;}") == {"outer": {"a": {"x": 1}, "b": 2}}
    parsed = ParsedFile(b"a {x 1;}; b 2;")
    assert parsed[("a", "x")] == 1
    assert parsed[("b",)] == 2


def test_spurious_semicolon_after_standalone_list() -> None:
    # Same getKeyword discard; parcelInjectionProperties files end with ");"
    parsed = ParsedFile(b"(\n (0 1 2) (3 4 5) 1.0 2.0 (1)\n);")
    assert parsed[()] == [[0, 1, 2], [3, 4, 5], 1.0, 2.0, [1]]


@pytest.mark.xfail
def test_value_then_subdictionary() -> None:
    # primitiveEntry::read reads tokens until a ';' at blockCount 0, so a
    # trailing subdictionary is part of the value (fvSchemes ddtSchemes/
    # divSchemes).
    assert _file_dict(
        b"ddtSchemes {default CrankNicolson ocCoeff {type scale; value 0.9;};}"
    ) == {
        "ddtSchemes": {
            "default": (
                ("CrankNicolson", "ocCoeff"),
                {"type": "scale", "value": 0.9},
            )
        }
    }
    assert _file_dict(b"key CrankNicolson {x 1;};") == {
        "key": ("CrankNicolson", {"x": 1})
    }
    assert _file_dict(
        b"div(phi,Yi_h) Gauss multivariateSelection {O2 limitedLinear01 1;};"
    ) == {
        "div(phi,Yi_h)": (
            ("Gauss", "multivariateSelection"),
            {"O2": ("limitedLinear01", 1)},
        )
    }


def test_include_func_args_on_next_line() -> None:
    # functionEntry::readFuncNameArgs reads the next token in case the
    # optional arguments start on the next line; name and args are stored as
    # one value.
    parsed = ParsedFile(
        b"functions {\n#includeFunc streamlinesSphere\n"
        b"( name=streamlines, fields=(U) )\n}"
    )
    assert parsed[("functions", "#includeFunc")] == (
        "streamlinesSphere( name=streamlines, fields=(U) )"
    )
    # Same-line arguments are unchanged.
    assert _file_dict(b"functions {#includeFunc fieldAverage(U, p)\n}") == {
        "functions": {"#includeFunc": "fieldAverage(U, p)"}
    }
    # Arguments may contain dimension sets and nested lists.
    assert _file_dict(
        b"functions {#includeFunc uniform( dimensions = [0 0 0 0 0 0 0], value = 0.5 )\n}"
    ) == {
        "functions": {
            "#includeFunc": ("uniform( dimensions = [0 0 0 0 0 0 0], value = 0.5 )")
        }
    }
    # Other directives still require the value on one line.
    assert _file_dict(b'#include "file"\n') == {"#include": '"file"'}


def test_stray_top_level_brace() -> None:
    # dictionary::read loops until entry::New returns false; an END_BLOCK
    # token ends the read silently (wingMotion/.../system/controlDict).
    assert _file_dict(b"a 1;\n}") == {"a": 1}
    assert _file_dict(b"a 1;\n}\n// footer\n") == {"a": 1}
    parsed = ParsedFile(b"a 1;\n}\n// footer\n")
    assert parsed[("a",)] == 1


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (b"[m^2 s^-3]", DimensionSet(length=2, time=-3)),
        (b"[m s^-1]", DimensionSet(length=1, time=-1)),
        (b"[kg m s^-2]", DimensionSet(mass=1, length=1, time=-2)),
        (b"[Pa]", DimensionSet(mass=1, length=-1, time=-2)),
        (b"[N m^-2]", DimensionSet(mass=1, length=-1, time=-2)),
        (b"[m^0.5]", DimensionSet(length=0.5)),
        # Explicit operators: '*' and '/' bind at priority 2, '^' at 3, and
        # parentheses group.
        (b"[kg*m^-3]", DimensionSet(mass=1, length=-3)),
        (b"[kg/m^3]", DimensionSet(mass=1, length=-3)),
        (b"[N/m^2]", DimensionSet(mass=1, length=-1, time=-2)),
        (b"[m^2/s]", DimensionSet(length=2, time=-1)),
        (b"[m/s/s]", DimensionSet(length=1, time=-2)),
        (b"[J/(kg K)]", DimensionSet(length=2, time=-2, temperature=-1)),
        (b"[kg/(m s^2)]", DimensionSet(mass=1, length=-1, time=-2)),
        (b"[(kg m)/s^2]", DimensionSet(mass=1, length=1, time=-2)),
    ],
)
def test_symbolic_dimensions(contents: bytes, expected: DimensionSet) -> None:
    # dimensionSet::read parses consecutive unit-symbol words with powers
    # from the default SI unit set.
    assert ParsedFile(b"dimensions " + contents + b";")[("dimensions",)] == expected


@pytest.mark.parametrize(
    "contents",
    [
        b"[foo]",
        b"[m^]",
        b"[m^x]",
        b"[m^-]",
        b"[m^++]",
        b"[m)]",
        b"[km^400]",
    ],
)
def test_symbolic_dimensions_invalid(contents: bytes) -> None:
    # FoamFileDecodeError is the parser's only error type; a non-numeric power
    # used to escape as a bare ValueError from float().
    with pytest.raises(FoamFileDecodeError):
        ParsedFile(b"dimensions " + contents + b";")


def test_named_dimensions_still_accepted() -> None:
    assert ParsedFile(b"dimensions [velocity];")[("dimensions",)] == DimensionSet(
        length=1, time=-1
    )


# etc/controlDict DimensionSets/SICoeffs: the only unit symbols whose scale
# multiplier is not 1.
SCALED_UNITS = {"cm": 1e-2, "mm": 1e-3, "km": 1e3}


@pytest.mark.parametrize("symbol", sorted(SCALED_UNITS))
def test_scaled_units_rejected_in_dimension_set(symbol: str) -> None:
    # operator>>(Istream&, dimensionSet&): "Cannot use scaled units in
    # dimensionSet". Accepting them silently discarded the scale.
    for contents in (
        f"[{symbol}]".encode(),
        f"[{symbol} s^-1]".encode(),
        f"[{symbol}^2]".encode(),
    ):
        with pytest.raises(FoamFileDecodeError, match="unscaled units"):
            ParsedFile(b"dimensions " + contents + b";")


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        (b"m", 5.0),
        (b"Pa", 5.0),
        (b"mm km", 5.0),  # 1e-3 * 1e3 == 1
        (b"cm", 5e-2),
        (b"mm", 5e-3),
        (b"km", 5e3),
        (b"cm^2", 5e-4),
        (b"cm^-1", 5e2),
        (b"km^3", 5e9),
        (b"cm s^-1", 5e-2),
        (b"km/s", 5e3),
        (b"m/mm", 5e3),
        (b"kg/cm^3", 5e6),
    ],
)
def test_dimensioned_value_scaled_by_units(units: bytes, expected: float) -> None:
    # dimensioned<Type>::initialise reads the dimension set and its multiplier,
    # then applies the multiplier to the value: "p [cm] 5" is 5 cm, i.e. 0.05 m.
    value = ParsedFile(b"p p [" + units + b"] 5;")[("p",)]
    assert isinstance(value, Dimensioned)
    assert value.value == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_dimensioned_tensor_value_scaled_by_units() -> None:
    value = ParsedFile(b"U U [cm s^-1] (1 2 3);")[("U",)]
    assert isinstance(value, Dimensioned)
    assert value.dimensions == DimensionSet(length=1, time=-1)
    assert value.value == pytest.approx([1e-2, 2e-2, 3e-2], rel=1e-12, abs=0.0)


def test_dimensioned_value_with_symbolic_units() -> None:
    parsed = ParsedFile(b"p [Pa] 1e5;")

    value = parsed[("p",)]
    assert isinstance(value, Dimensioned)
    assert value.dimensions == DimensionSet(mass=1, length=-1, time=-2)
    assert value.value == 100000.0


def test_bare_dollar_reference_entry() -> None:
    # entry::New substitution entries ($name) replace the whole entry and
    # take no statement terminator.
    assert _file_dict(b"TiO2_s {$TiO2}") == {"TiO2_s": {"$TiO2": None}}
    assert _file_dict(b"TiO2_s {$TiO2;}") == {"TiO2_s": {"$TiO2": None}}
    parsed = ParsedFile(b"TiO2 {rho 2000;} TiO2_s {$TiO2}")
    assert parsed[("TiO2_s", "$TiO2")] is None
    assert FoamFile.dumps(_file_dict(b"TiO2_s {$TiO2}"), ensure_header=False) == (
        b"TiO2_s {$TiO2}"
    )


def test_grammar_tolerance_round_trip() -> None:
    cases = [
        b"a {x 1;}; b 2;",
        b"(\n (0 1 2) (3 4 5) 1.0 2.0 (1)\n);",
        # b"ddtSchemes {default CrankNicolson ocCoeff {type scale; value 0.9;};}",
        b"functions {#includeFunc fieldAverage(U, p)\n}",
        b"dimensions [m^2 s^-3];",
        b"TiO2_s {$TiO2}",
    ]
    for contents in cases:
        dumped = FoamFile.dumps(_file_dict(contents), ensure_header=False)
        assert _file_dict(dumped) == _file_dict(contents)
