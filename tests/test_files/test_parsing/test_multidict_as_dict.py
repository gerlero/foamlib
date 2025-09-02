"""
Test that Parsed.as_dict() returns MultiDicts instead of standard dicts.
"""

import pytest
from foamlib._files._parsing import Parsed
from multicollections import MultiDict


def test_as_dict_returns_multidict_simple():
    """Test that as_dict() returns MultiDict for simple case."""
    parsed = Parsed(b"""
        my_dict
        {
            version     2.0;
            format      ascii;
        }
    """)
    
    result = parsed.as_dict()
    
    # Check that the result is a MultiDict
    assert isinstance(result, MultiDict)
    
    # Check that nested dict is also a MultiDict
    nested = result['my_dict']
    assert isinstance(nested, MultiDict)
    
    # Verify content is still accessible
    assert nested['version'] == 2.0
    assert nested['format'] == "ascii"


def test_as_dict_returns_multidict_nested():
    """Test that as_dict() returns MultiDict for nested case."""
    parsed = Parsed(b"""
        my_nested_dict
        {
            p
            {
                solver            PCG;
                preconditioner    DIC;
            }
            U
            {
                solver       smoothSolver;
                smoother     symGaussSeidel;
            }
        }
    """)
    
    result = parsed.as_dict()
    
    # Check that the result is a MultiDict
    assert isinstance(result, MultiDict)
    
    # Check that first level nested dict is also a MultiDict
    nested = result['my_nested_dict']
    assert isinstance(nested, MultiDict)
    
    # Check that second level nested dicts are also MultiDicts
    p_dict = nested['p']
    assert isinstance(p_dict, MultiDict)
    
    u_dict = nested['U']
    assert isinstance(u_dict, MultiDict)
    
    # Verify content is still accessible
    assert p_dict['solver'] == "PCG"
    assert p_dict['preconditioner'] == "DIC"
    assert u_dict['solver'] == "smoothSolver"
    assert u_dict['smoother'] == "symGaussSeidel"


def test_as_dict_returns_multidict_empty():
    """Test that as_dict() returns MultiDict for empty dict case."""
    parsed = Parsed(b"""
        empty_dict
        {
        }
    """)
    
    result = parsed.as_dict()
    
    # Check that the result is a MultiDict
    assert isinstance(result, MultiDict)
    
    # Check that empty nested dict is also a MultiDict
    nested = result['empty_dict']
    assert isinstance(nested, MultiDict)
    assert len(nested) == 0


def test_as_dict_returns_multidict_with_standalone_data():
    """Test that as_dict() returns MultiDict when there's standalone data."""
    parsed = Parsed(b"""
        someValue 42;
        someDict
        {
            key value;
        }
    """)
    
    result = parsed.as_dict()
    
    # Check that the result is a MultiDict
    assert isinstance(result, MultiDict)
    
    # Check the standalone data
    assert result['someValue'] == 42
    
    # Check that nested dict is also a MultiDict
    nested = result['someDict']
    assert isinstance(nested, MultiDict)
    assert nested['key'] == "value"


def test_as_dict_multidict_behavior():
    """Test that the returned MultiDict behaves as expected."""
    parsed = Parsed(b"""
        singleKey value1;
        dict
        {
            innerKey value;
        }
    """)
    
    result = parsed.as_dict()
    
    # Check that it's a MultiDict
    assert isinstance(result, MultiDict)
    
    # Regular access should work
    assert result['singleKey'] == 'value1'
    
    # Nested dict should be MultiDict
    nested = result['dict']
    assert isinstance(nested, MultiDict)
    assert nested['innerKey'] == "value"
    
    # MultiDict should have the expected methods
    assert hasattr(result, 'getall')
    assert hasattr(result, 'add')
    
    # Should behave like a normal dict for single values
    assert result.get('singleKey') == 'value1'
    assert result.get('nonexistent') is None