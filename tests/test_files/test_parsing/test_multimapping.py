"""Test MultiMapping behavior of Parsed and FoamFile classes."""

import tempfile
import os

import pytest

from foamlib._files._parsing import Parsed
from foamlib._files._files import FoamFile


def test_parsed_multimapping_methods():
    """Test MultiMapping methods for Parsed class."""
    parsed = Parsed(b'test 1; test2 "value2";')
    
    # Test getall method
    assert parsed.getall(("test",)) == [1]
    assert parsed.getall(("test2",)) == ['"value2"']
    
    # Test getall with nonexistent key raises KeyError
    with pytest.raises(KeyError):
        parsed.getall(("nonexistent",))
    
    # Test getone method  
    assert parsed.getone(("test",)) == 1
    assert parsed.getone(("test2",)) == '"value2"'
    
    # Test add method
    parsed.add(("test3",), "value3")
    assert parsed.getone(("test3",)) == "value3"
    
    # Test popone method
    value = parsed.popone(("test3",))
    assert value == "value3"
    assert ("test3",) not in parsed


def test_foamfile_multimapping_methods():
    """Test MultiMapping methods for FoamFile class."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.foam') as f:
        f.write('test 1;\ntest2 "value2";\n')
        temp_path = f.name
    
    try:
        foam_file = FoamFile(temp_path)
        
        # Test getall method
        assert foam_file.getall(("test",)) == [1]
        assert foam_file.getall(("test2",)) == ['"value2"']
        
        # Test getall with nonexistent key raises KeyError
        with pytest.raises(KeyError):
            foam_file.getall(("nonexistent",))
        
        # Test getone method
        assert foam_file.getone(("test",)) == 1
        assert foam_file.getone(("test2",)) == '"value2"'
        
        # Test add method
        foam_file.add(("test3",), "value3")
        assert foam_file.getone(("test3",)) == "value3"
        
        # Test popone method
        value = foam_file.popone(("test3",))
        assert value == "value3"
        assert ("test3",) not in foam_file
        
    finally:
        os.unlink(temp_path)


def test_multimapping_backward_compatibility():
    """Test that existing API still works with MultiMapping inheritance."""
    # Test Parsed
    parsed = Parsed(b'test 1; test2 "value2";')
    assert parsed[("test",)] == 1
    assert parsed[("test2",)] == '"value2"'
    assert len(parsed) == 2
    assert ("test",) in parsed
    assert ("nonexistent",) not in parsed
    
    # Test iteration still works
    keys = list(parsed.keys())
    assert ("test",) in keys
    assert ("test2",) in keys
    
    # Test FoamFile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.foam') as f:
        f.write('test 1;\ntest2 "value2";\n')
        temp_path = f.name
    
    try:
        foam_file = FoamFile(temp_path)
        assert foam_file[("test",)] == 1
        assert foam_file[("test2",)] == '"value2"'
        assert len(foam_file) == 2
        assert ("test",) in foam_file
        assert ("nonexistent",) not in foam_file
        
        # Test iteration still works
        keys = list(foam_file.keys())
        assert "test" in keys
        assert "test2" in keys
        
    finally:
        os.unlink(temp_path)