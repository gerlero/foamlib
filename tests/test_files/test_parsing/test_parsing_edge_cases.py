"""Test additional parsing edge cases and complex scenarios."""

import pytest
from foamlib._files._parsing import Parsed


def test_parsing_mixed_comments():
    """Test parsing with mixed comment styles."""
    parsed = Parsed(b"""
        // Single line comment
        key1 value1;
        
        /* Multi-line
           comment spanning
           multiple lines */
        key2 value2;
        
        key3 /* inline comment */ value3;
        
        // Comment with special chars: // /* */ 
        key4 value4;
        
        /*
         * C-style comment block
         * with asterisks
         */
        key5 value5;
    """)
    
    assert parsed[("key1",)] == "value1"
    assert parsed[("key2",)] == "value2"
    assert parsed[("key3",)] == "value3"
    assert parsed[("key4",)] == "value4"
    assert parsed[("key5",)] == "value5"


def test_parsing_nested_parentheses():
    """Test parsing with deeply nested parentheses."""
    parsed = Parsed(b"""
        complexFunction ((a+b)*(c+d)+(e*(f+g)));
        nestedLists (((1 2) (3 4)) ((5 6) (7 8)));
        mathExpression ((sin(x)*cos(y))+(tan(z)));
    """)
    
    assert parsed[("complexFunction",)] == "((a+b)*(c+d)+(e*(f+g)))"
    assert parsed[("nestedLists",)] == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert parsed[("mathExpression",)] == "((sin(x)*cos(y))+(tan(z)))"


def test_parsing_scientific_notation():
    """Test parsing various scientific notation formats."""
    parsed = Parsed(b"""
        smallNumber 1.5e-15;
        largeNumber 2.3E+20;
        negativeExp -4.7e-8;
        positiveExp 6.1E+12;
        integerExp 5e10;
        decimal 1.234567890123;
        fraction 0.00001;
    """)
    
    assert parsed[("smallNumber",)] == 1.5e-15
    assert parsed[("largeNumber",)] == 2.3e20
    assert parsed[("negativeExp",)] == -4.7e-8
    assert parsed[("positiveExp",)] == 6.1e12
    assert parsed[("integerExp",)] == 5e10
    assert parsed[("decimal",)] == 1.234567890123
    assert parsed[("fraction",)] == 1e-5


def test_parsing_complex_strings():
    """Test parsing complex string patterns."""
    parsed = Parsed(b"""
        quotedString "string with spaces and symbols !@#$%^&*()";
        pathString "/home/user/foam/case with spaces/system";
        urlString "https://www.openfoam.org/documentation";
        regexString ".*\\.foam$";
        escapedQuotes "string with \\"escaped\\" quotes";
        multiWordKey "key with spaces" value;
    """)
    
    assert parsed[("quotedString",)] == "string with spaces and symbols !@#$%^&*()"
    assert parsed[("pathString",)] == "/home/user/foam/case with spaces/system"
    assert parsed[("urlString",)] == "https://www.openfoam.org/documentation"
    assert parsed[("regexString",)] == ".*\\.foam$"
    assert parsed[("escapedQuotes",)] == 'string with "escaped" quotes'
    assert parsed[("key with spaces",)] == "value"


def test_parsing_boolean_variations():
    """Test parsing various boolean representations."""
    parsed = Parsed(b"""
        boolTrue true;
        boolFalse false;
        boolOn on;
        boolOff off;
        boolYes yes;
        boolNo no;
        boolY y;
        boolN n;
        bool1 1;
        bool0 0;
    """)
    
    assert parsed[("boolTrue",)] is True
    assert parsed[("boolFalse",)] is False
    assert parsed[("boolOn",)] is True
    assert parsed[("boolOff",)] is False
    assert parsed[("boolYes",)] is True
    assert parsed[("boolNo",)] is False
    assert parsed[("boolY",)] is True
    assert parsed[("boolN",)] is False
    assert parsed[("bool1",)] == 1
    assert parsed[("bool0",)] == 0


def test_parsing_dimension_sets_variations():
    """Test parsing various dimension set formats."""
    parsed = Parsed(b"""
        dimStandard [0 1 -2 0 0 0 0];
        dimSpaced [ 1 1 -2 0 0 0 0 ];
        dimNegative [-1 -1 2 0 0 0 0];
        dimMixed [0.5 1 -2 0 0 0 0];
        dimWithComments [0 1 -2 /* pressure */ 0 0 0 0];
    """)
    
    from foamlib._files._types import DimensionSet
    
    assert parsed[("dimStandard",)] == DimensionSet(length=1, time=-2)
    assert parsed[("dimSpaced",)] == DimensionSet(mass=1, length=1, time=-2)
    assert parsed[("dimNegative",)] == DimensionSet(mass=-1, length=-1, time=2)


def test_parsing_complex_lists():
    """Test parsing complex list structures."""
    parsed = Parsed(b"""
        mixedList (1 "string" 2.5 true [0 1 -2 0 0 0 0]);
        listOfLists ((1 2 3) (4 5 6) (7 8 9));
        listOfDicts ({a 1; b 2;} {c 3; d 4;});
        nestedMixed (
            {
                sublist (1 2 3);
                subdict {x 10; y 20;};
            }
            (4 5 6)
            "standalone"
        );
    """)
    
    mixed_list = parsed[("mixedList",)]
    assert mixed_list[0] == 1
    assert mixed_list[1] == "string"
    assert mixed_list[2] == 2.5
    assert mixed_list[3] is True
    
    list_of_lists = parsed[("listOfLists",)]
    assert list_of_lists == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_parsing_openfoam_specific_syntax():
    """Test parsing OpenFOAM-specific syntax patterns."""
    parsed = Parsed(b"""
        divSchemes
        {
            default         none;
            div(phi,U)      Gauss linearUpwind grad(U);
            div(phi,k)      Gauss upwind;
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
        }
        
        laplacianSchemes
        {
            default         Gauss linear corrected;
            laplacian((rho*nuEff),U) Gauss linear corrected;
        }
        
        interpolationSchemes
        {
            default         linear;
            interpolate(rho*U) linear;
        }
    """)
    
    assert parsed[("divSchemes", "div(phi,U)")] == ("Gauss", "linearUpwind", "grad(U)")
    assert parsed[("divSchemes", "div((nuEff*dev2(T(grad(U)))))")] == ("Gauss", "linear")
    assert parsed[("laplacianSchemes", "laplacian((rho*nuEff),U)")] == ("Gauss", "linear", "corrected")


def test_parsing_macros_and_variables():
    """Test parsing macros and variable references."""
    parsed = Parsed(b"""
        myVar 42;
        myRef $myVar;
        pathRef $FOAM_CASE/constant;
        nestedRef ${myVar};
        complexRef ${FOAM_CASE}/system;
        mathRef #eval{$myVar * 2};
        
        // Macro definitions
        defineMyMacro
        {
            value $myVar;
            multiplied #eval{$myVar * 3};
        }
    """)
    
    assert parsed[("myVar",)] == 42
    assert parsed[("myRef",)] == "$myVar"
    assert parsed[("pathRef",)] == "$FOAM_CASE/constant"
    assert parsed[("nestedRef",)] == "${myVar}"
    assert parsed[("mathRef",)] == "#eval{$myVar * 2}"


def test_parsing_special_characters_in_values():
    """Test parsing values with special characters."""
    parsed = Parsed(b"""
        pathWithSpaces "/path with spaces/file.txt";
        regexPattern ".*\\.(foam|OF)$";
        mathExpression "sin(2*pi*x/L)";
        shellCommand "find . -name '*.C' | wc -l";
        urlPath "http://example.com/path?param=value&other=123";
        specialChars "!@#$%^&*()_+-=[]{}|;':,.<>?";
    """)
    
    assert parsed[("pathWithSpaces",)] == "/path with spaces/file.txt"
    assert parsed[("regexPattern",)] == ".*\\.(foam|OF)$"
    assert parsed[("mathExpression",)] == "sin(2*pi*x/L)"
    assert parsed[("shellCommand",)] == "find . -name '*.C' | wc -l"
    assert parsed[("urlPath",)] == "http://example.com/path?param=value&other=123"
    assert parsed[("specialChars",)] == "!@#$%^&*()_+-=[]{}|;':,.<>?"


def test_parsing_whitespace_handling():
    """Test parsing with various whitespace patterns."""
    parsed = Parsed(b"""
        // Test various spacing
        key1    value1;
        key2	value2;  // tab
        key3
            value3;  // newline
        key4		value4;    // mixed tabs/spaces
        
        listWithSpaces
        (
            item1
            item2    item3
            item4	item5
        );
        
        dictWithSpacing
        {
            key1     value1;
            key2 value2;
            key3	value3;
        }
    """)
    
    assert parsed[("key1",)] == "value1"
    assert parsed[("key2",)] == "value2"
    assert parsed[("key3",)] == "value3"
    assert parsed[("key4",)] == "value4"
    assert parsed[("listWithSpaces",)] == ["item1", "item2", "item3", "item4", "item5"]


def test_parsing_edge_case_numbers():
    """Test parsing edge case number formats."""
    parsed = Parsed(b"""
        leadingZeros 007;
        trailingZeros 123.000;
        noLeadingZero .5;
        noTrailingZero 5.;
        signedPositive +42;
        signedNegative -42;
        infinity inf;
        negativeInf -inf;
        notANumber nan;
    """)
    
    assert parsed[("leadingZeros",)] == 7
    assert parsed[("trailingZeros",)] == 123.0
    assert parsed[("noLeadingZero",)] == 0.5
    assert parsed[("noTrailingZero",)] == 5.0
    assert parsed[("signedPositive",)] == 42
    assert parsed[("signedNegative",)] == -42


def test_parsing_empty_structures():
    """Test parsing empty dictionaries and lists."""
    parsed = Parsed(b"""
        emptyDict {};
        emptyList ();
        emptyDictWithSpaces {   };
        emptyListWithSpaces (   );
        emptyDictWithNewlines
        {
        };
        emptyListWithNewlines
        (
        );
    """)
    
    assert parsed[("emptyDict",)] == {}
    assert parsed[("emptyList",)] == []
    assert parsed[("emptyDictWithSpaces",)] == {}
    assert parsed[("emptyListWithSpaces",)] == []
    assert parsed[("emptyDictWithNewlines",)] == {}
    assert parsed[("emptyListWithNewlines",)] == []


def test_parsing_boundary_field_syntax():
    """Test parsing boundary field syntax."""
    parsed = Parsed(b"""
        boundaryField
        {
            inlet
            {
                type            fixedValue;
                value           uniform (1 0 0);
            }
            
            outlet
            {
                type            zeroGradient;
            }
            
            walls
            {
                type            noSlip;
            }
            
            "movingWall.*"
            {
                type            movingWallVelocity;
                value           uniform (0 1 0);
            }
        }
    """)
    
    assert parsed[("boundaryField", "inlet", "type")] == "fixedValue"
    assert parsed[("boundaryField", "inlet", "value")] == ("uniform", [1, 0, 0])
    assert parsed[("boundaryField", "outlet", "type")] == "zeroGradient"
    assert parsed[("boundaryField", "movingWall.*", "type")] == "movingWallVelocity"


def test_parsing_function_object_syntax():
    """Test parsing function object syntax."""
    parsed = Parsed(b"""
        functions
        {
            probes
            {
                type            probes;
                libs            ("libsampling.so");
                writeControl    timeStep;
                writeInterval   1;
                
                fields
                (
                    U
                    p
                );
                
                probeLocations
                (
                    (1 0.5 0.5)
                    (2 0.5 0.5)
                    (3 0.5 0.5)
                );
            }
            
            forces
            {
                type            forces;
                libs            ("libforces.so");
                patches         (cylinder);
                rho             rhoInf;
                rhoInf          1;
                CofR            (0 0 0);
            }
        }
    """)
    
    assert parsed[("functions", "probes", "type")] == "probes"
    assert parsed[("functions", "probes", "fields")] == ["U", "p"]
    assert parsed[("functions", "probes", "probeLocations")] == [[1, 0.5, 0.5], [2, 0.5, 0.5], [3, 0.5, 0.5]]
    assert parsed[("functions", "forces", "patches")] == ["cylinder"]