"""Test MultiMapping functionality for Parsed and FoamFile classes."""

import tempfile
import os
from pathlib import Path

import pytest

from foamlib._files._parsing import Parsed
from foamlib._files._files import FoamFile


class TestParsedMultiMapping:
    """Test MultiMapping functionality for Parsed class."""

    def test_basic_multimapping_behavior(self):
        """Test basic MultiMapping interface methods."""
        sample_content = b'''
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

key1        value1;
key2        value2;
key1        value3;
key3        value4;
'''
        parsed = Parsed(sample_content)

        # Test getone - should return first value
        assert parsed.getone(('key1',)) == 'value1'
        assert parsed[('key1',)] == 'value1'  # __getitem__ should be same as getone

        # Test getall - should return all values
        assert parsed.getall(('key1',)) == ['value1', 'value3']
        assert parsed.getall(('key2',)) == ['value2']
        assert parsed.getall(('key3',)) == ['value4']

        # Test keys with duplicates
        keys = list(parsed.keys())
        assert keys.count(('key1',)) == 2  # key1 appears twice
        assert keys.count(('key2',)) == 1
        assert keys.count(('key3',)) == 1

        # Test length - should be total number of key-value pairs
        assert len(parsed) == 9  # FoamFile + 4 FoamFile entries + 4 regular entries (key1 twice) = 9 total

    def test_single_values(self):
        """Test behavior with single values (no duplicates)."""
        sample_content = b'''
FoamFile
{
    version     2.0;
    format      ascii;
}

key1        value1;
key2        value2;
'''
        parsed = Parsed(sample_content)

        assert parsed.getone(('key1',)) == 'value1'
        assert parsed.getall(('key1',)) == ['value1']
        assert parsed[('key1',)] == 'value1'

    def test_missing_key(self):
        """Test behavior with missing keys."""
        sample_content = b'''
FoamFile
{
    version     2.0;
}

key1        value1;
'''
        parsed = Parsed(sample_content)

        with pytest.raises(KeyError):
            parsed[('missing',)]

        with pytest.raises(KeyError):
            parsed.getone(('missing',))

        with pytest.raises(KeyError):
            parsed.getall(('missing',))

    def test_nested_keys(self):
        """Test MultiMapping with nested keys."""
        sample_content = b'''
FoamFile
{
    version     2.0;
}

dict1
{
    key1        value1;
    key1        value2;
}
'''
        parsed = Parsed(sample_content)

        # Test nested keys with duplicates
        assert parsed.getall(('dict1', 'key1')) == ['value1', 'value2']
        assert parsed.getone(('dict1', 'key1')) == 'value1'


class TestFoamFileMultiMapping:
    """Test MultiMapping functionality for FoamFile class."""

    def setup_method(self):
        """Create a temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.foam', delete=False)
        self.temp_file.write('''
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

key1        value1;
key2        value2;
key1        value3;
key3        value4;
''')
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def teardown_method(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)

    def test_basic_multimapping_behavior(self):
        """Test basic MultiMapping interface methods for FoamFile."""
        foam_file = FoamFile(self.temp_path)

        # Test getone - should return first value
        assert foam_file.getone('key1') == 'value1'
        assert foam_file['key1'] == 'value1'  # __getitem__ should be same as getone

        # Test getall - should return all values
        assert foam_file.getall('key1') == ['value1', 'value3']
        assert foam_file.getall('key2') == ['value2']
        assert foam_file.getall('key3') == ['value4']

        # Test that we can access regular entries
        assert foam_file['key2'] == 'value2'
        assert foam_file['key3'] == 'value4'

    def test_multimapping_with_tuple_keys(self):
        """Test MultiMapping with tuple keys."""
        foam_file = FoamFile(self.temp_path)

        # Test nested access (if any)
        foam_file_version = foam_file.getone(('FoamFile', 'version'))
        assert foam_file_version == 2.0

    def test_multimapping_operations(self):
        """Test MultiMapping operations like add and popone."""
        foam_file = FoamFile(self.temp_path)

        # Test popone
        popped_value = foam_file.popone('key3')
        assert popped_value == 'value4'
        
        # After popping, key3 should not exist
        with pytest.raises(KeyError):
            foam_file['key3']

    def test_multimapping_with_missing_keys(self):
        """Test behavior with missing keys."""
        foam_file = FoamFile(self.temp_path)

        with pytest.raises(KeyError):
            foam_file.getone('missing')

        with pytest.raises(KeyError):
            foam_file.getall('missing')

        with pytest.raises(KeyError):
            foam_file.popone('missing')

    def test_add_method(self):
        """Test the add method (currently delegates to __setitem__)."""
        foam_file = FoamFile(self.temp_path)

        # Add a new key-value pair
        foam_file.add('new_key', 'new_value')
        assert foam_file['new_key'] == 'new_value'
        assert foam_file.getall('new_key') == ['new_value']


class TestMultiMappingBackwardsCompatibility:
    """Test that MultiMapping changes don't break existing functionality."""

    def test_existing_behavior_preserved(self):
        """Test that existing single-value behavior is preserved."""
        sample_content = b'''
FoamFile
{
    version     2.0;
    format      ascii;
}

key1        value1;
key2        value2;
'''
        parsed = Parsed(sample_content)

        # Standard Mapping interface should still work
        assert parsed[('key1',)] == 'value1'
        assert parsed[('key2',)] == 'value2'
        assert ('key1',) in parsed
        assert ('missing',) not in parsed
        assert len(list(parsed.keys())) == 5  # FoamFile + version + format + key1 + key2

    def test_file_operations_preserved(self):
        """Test that file operations work normally."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.foam', delete=False) as f:
            f.write('''
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
}

key1        value1;
key2        value2;
''')
            temp_path = f.name

        try:
            foam_file = FoamFile(temp_path)
            
            # Test standard operations
            assert foam_file['key1'] == 'value1'
            foam_file['key3'] = 'value3'
            assert foam_file['key3'] == 'value3'
            
            del foam_file['key2']
            with pytest.raises(KeyError):
                foam_file['key2']
                
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])