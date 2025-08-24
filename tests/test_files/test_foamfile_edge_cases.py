"""Test edge cases and error handling for FoamFile."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from foamlib import FoamFile
from foamlib._files._parsing import Parsed


def test_foamfile_property_type_errors():
    """Test that FoamFile properties raise appropriate type errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Set up a minimal file structure
        ff["FoamFile"] = {
            "version": "not_a_number",
            "format": "invalid_format", 
            "class": 123,
            "location": ["not", "a", "string"],
            "object": {"not": "a_string"}
        }
        
        # Test version property type error
        with pytest.raises(TypeError, match="version is not a number"):
            _ = ff.version
            
        # Test format property type error  
        with pytest.raises(TypeError, match="format is not a string"):
            _ = ff.format
            
        # Test class property type error
        with pytest.raises(TypeError, match="class is not a string"):
            _ = ff.class_
            
        # Test location property type error
        with pytest.raises(TypeError, match="location is not a string"):
            _ = ff.location
            
        # Test object property type error
        with pytest.raises(TypeError, match="object is not a string"):
            _ = ff.object_


def test_foamfile_format_value_error():
    """Test that FoamFile format property raises ValueError for invalid values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        ff["FoamFile"] = {"format": "invalid"}
        
        with pytest.raises(ValueError, match="format is not 'ascii' or 'binary'"):
            _ = ff.format


def test_foamfile_context_manager_exception_handling():
    """Test FoamFile context manager behavior when exceptions occur."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Test that changes are still saved when exception occurs in context
        try:
            with ff:
                ff["key1"] = "value1"
                ff["key2"] = "value2"
                raise ValueError("Test exception")
        except ValueError:
            pass
            
        # Verify changes were saved despite exception
        assert ff["key1"] == "value1"
        assert ff["key2"] == "value2"


def test_foamfile_deep_nesting():
    """Test FoamFile with deeply nested dictionaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Create deeply nested structure
        ff["level1"] = {"level2": {"level3": {"level4": {"value": 42}}}}
        
        # Test deep access
        assert ff["level1", "level2", "level3", "level4", "value"] == 42
        
        # Test SubDict chaining
        subdict = ff["level1"]["level2"]["level3"]
        assert subdict["level4", "value"] == 42
        
        # Test modification at deep level
        ff["level1", "level2", "level3", "level4", "new_value"] = "test"
        assert ff["level1", "level2", "level3", "level4", "new_value"] == "test"


def test_foamfile_empty_and_none_keys():
    """Test FoamFile handling of empty and None keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Test None key (standalone data)
        ff[None] = "standalone_value"
        assert ff[None] == "standalone_value"
        
        # Test empty tuple key
        ff[()] = "empty_tuple_value"
        assert ff[()] == "empty_tuple_value"
        
        # Test that None and () refer to the same data
        ff[None] = "updated_value"
        assert ff[()] == "updated_value"


def test_foamfile_special_characters_in_keys():
    """Test FoamFile with special characters in keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Keys with special characters that are valid in OpenFOAM
        special_keys = [
            "div(phi,U)",
            "laplacian(nu,U)",
            "grad(p)",
            "ddt(rho)",
            "Re_tau",
            "omega.air",
            "U_0"
        ]
        
        for key in special_keys:
            ff[key] = f"value_for_{key}"
            assert ff[key] == f"value_for_{key}"


def test_foamfile_update_and_clear():
    """Test FoamFile update and clear methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Initial data
        ff["key1"] = "value1"
        ff["key2"] = "value2"
        
        # Test update with dict
        ff.update({"key2": "updated_value2", "key3": "value3"})
        assert ff["key1"] == "value1"
        assert ff["key2"] == "updated_value2"
        assert ff["key3"] == "value3"
        
        # Test update with keyword arguments
        ff.update(key4="value4", key5="value5")
        assert ff["key4"] == "value4"
        assert ff["key5"] == "value5"
        
        # Test clear
        initial_len = len(ff)
        assert initial_len > 0
        ff.clear()
        # Should only have FoamFile header remaining
        assert len(ff) == 0
        assert "FoamFile" in ff._get_parsed()


def test_foamfile_iteration_and_contains():
    """Test FoamFile iteration and membership testing.""" 
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Add some test data
        ff["key1"] = "value1"
        ff["key2"] = {"subkey": "subvalue"}
        ff[None] = "standalone"
        
        # Test iteration (should exclude FoamFile header)
        keys = list(ff)
        assert "key1" in keys
        assert "key2" in keys
        assert None in keys
        assert "FoamFile" not in keys
        
        # Test contains
        assert "key1" in ff
        assert "key2" in ff
        assert None in ff
        assert ("key2", "subkey") in ff
        assert "nonexistent" not in ff
        assert ("key2", "nonexistent") not in ff


def test_foamfile_as_dict():
    """Test FoamFile as_dict method with various data structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Set up complex nested structure
        ff["simple"] = "value"
        ff["list"] = [1, 2, 3]
        ff["nested"] = {
            "sub1": "subvalue1",
            "sub2": {"deep": "deepvalue"}
        }
        
        as_dict = ff.as_dict()
        
        # Verify structure
        assert as_dict["simple"] == "value"
        assert as_dict["list"] == [1, 2, 3]
        assert as_dict["nested"]["sub1"] == "subvalue1"
        assert as_dict["nested"]["sub2"]["deep"] == "deepvalue"
        
        # Verify FoamFile header is included
        assert "FoamFile" in as_dict
        assert "version" in as_dict["FoamFile"]


def test_foamfile_nonexistent_file_access():
    """Test accessing entries in a non-existent file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "nonexistent.dict"
        ff = FoamFile(test_file)
        
        # Should raise FileNotFoundError when trying to access
        with pytest.raises(FileNotFoundError):
            _ = ff["key"]
            
        # Should raise FileNotFoundError when using context manager
        with pytest.raises(FileNotFoundError):
            with ff:
                _ = ff["key"]


def test_foamfile_path_property():
    """Test FoamFile path property and __fspath__ method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        assert ff.path == test_file
        assert str(ff) == str(test_file)
        assert ff.__fspath__() == str(test_file)


def test_foamfile_repr():
    """Test FoamFile string representation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict" 
        ff = FoamFile(test_file)
        
        repr_str = repr(ff)
        assert "FoamFile" in repr_str
        assert str(test_file) in repr_str


def test_subdict_edge_cases():
    """Test FoamFile.SubDict edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Create nested structure
        ff["outer"] = {"inner": {"value": 42}}
        
        # Get SubDict
        subdict = ff["outer"]
        assert isinstance(subdict, FoamFile.SubDict)
        
        # Test SubDict properties
        assert subdict._file is ff
        assert subdict._keywords == ("outer",)
        
        # Test deep SubDict access
        inner_subdict = subdict["inner"]
        assert isinstance(inner_subdict, FoamFile.SubDict)
        assert inner_subdict._keywords == ("outer", "inner")
        
        # Test SubDict iteration
        keys = list(subdict)
        assert "inner" in keys
        
        # Test SubDict as_dict
        subdict_as_dict = subdict.as_dict()
        assert subdict_as_dict["inner"]["value"] == 42


def test_foamfile_error_recovery():
    """Test FoamFile behavior when file operations fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        ff = FoamFile(test_file)
        
        # Create initial data
        ff["key"] = "value"
        
        # Simulate file being deleted between operations
        test_file.unlink()
        
        # Should still be able to access cached data
        assert ff["key"] == "value"
        
        # Should create new file when setting new data
        ff["new_key"] = "new_value"
        assert test_file.exists()
        assert ff["new_key"] == "new_value"


def test_foamfile_with_parsing_edge_cases():
    """Test FoamFile with edge cases in parsing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.dict"
        
        # Create file with edge case content
        content = b"""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
        }
        
        // Comments should be preserved
        /* Multi-line
           comments too */
           
        emptyDict
        {
        }
        
        listWithDict
        (
            {
                key value;
            }
            {
                key2 value2;  
            }
        );
        
        #include "someFile"
        
        $macro expansion;
        """
        
        test_file.write_bytes(content)
        ff = FoamFile(test_file)
        
        # Should be able to read the file
        assert "emptyDict" in ff
        assert "listWithDict" in ff
        assert "#include" in ff
        assert "$macro" in ff
        
        # Empty dict should be accessible
        empty_dict = ff["emptyDict"]
        assert isinstance(empty_dict, FoamFile.SubDict)
        assert len(empty_dict) == 0