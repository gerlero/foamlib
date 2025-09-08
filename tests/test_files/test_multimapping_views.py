"""Test that FoamFile and SubDict methods return MultiMappingView subclasses."""

import tempfile
from pathlib import Path

import multicollections.abc
import pytest

from foamlib import FoamFile


@pytest.fixture
def sample_foam_file():
    """Create a temporary FoamFile for testing."""
    content = b"""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     simpleFoam;
startFrom       startTime;
startTime       0;

sub1
{
    key1 value1;
    key2 value2;
}

sub2
{
    key3 value3;
    key4 value4;
}
"""
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.foam') as f:
        f.write(content)
        temp_file = Path(f.name)
    
    yield FoamFile(temp_file)
    
    # Cleanup
    temp_file.unlink()


def test_foam_file_keys_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.keys() returns a MultiMappingView subclass."""
    keys_view = sample_foam_file.keys()
    
    # Check type hierarchy
    assert isinstance(keys_view, multicollections.abc.MultiMappingView)
    assert isinstance(keys_view, multicollections.abc.KeysView)
    
    # Check functionality
    assert hasattr(keys_view, '__iter__')
    assert hasattr(keys_view, '__len__')
    assert hasattr(keys_view, '__contains__')
    
    # Test actual behavior
    keys_list = list(keys_view)
    assert 'application' in keys_list
    assert 'sub1' in keys_list
    assert 'sub2' in keys_list
    assert len(keys_view) == len(keys_list)
    assert 'application' in keys_view


def test_foam_file_values_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.values() returns a MultiMappingView subclass."""
    values_view = sample_foam_file.values()
    
    # Check type hierarchy
    assert isinstance(values_view, multicollections.abc.MultiMappingView)
    assert isinstance(values_view, multicollections.abc.ValuesView)
    
    # Check functionality
    assert hasattr(values_view, '__iter__')
    assert hasattr(values_view, '__len__')
    assert hasattr(values_view, '__contains__')
    
    # Test actual behavior
    values_list = list(values_view)
    assert 'simpleFoam' in values_list
    assert len(values_view) == len(values_list)
    assert 'simpleFoam' in values_view


def test_foam_file_items_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.items() returns a MultiMappingView subclass."""
    items_view = sample_foam_file.items()
    
    # Check type hierarchy
    assert isinstance(items_view, multicollections.abc.MultiMappingView)
    assert isinstance(items_view, multicollections.abc.ItemsView)
    
    # Check functionality
    assert hasattr(items_view, '__iter__')
    assert hasattr(items_view, '__len__')
    assert hasattr(items_view, '__contains__')
    
    # Test actual behavior
    items_list = list(items_view)
    expected_item = ('application', 'simpleFoam')
    assert expected_item in items_list
    assert len(items_view) == len(items_list)
    assert expected_item in items_view


def test_subdict_keys_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.SubDict.keys() returns a MultiMappingView subclass."""
    sub1 = sample_foam_file['sub1']
    keys_view = sub1.keys()
    
    # Check type hierarchy
    assert isinstance(keys_view, multicollections.abc.MultiMappingView)
    assert isinstance(keys_view, multicollections.abc.KeysView)
    
    # Check functionality
    assert hasattr(keys_view, '__iter__')
    assert hasattr(keys_view, '__len__')
    assert hasattr(keys_view, '__contains__')
    
    # Test actual behavior
    keys_list = list(keys_view)
    assert 'key1' in keys_list
    assert 'key2' in keys_list
    assert len(keys_view) == 2
    assert 'key1' in keys_view


def test_subdict_values_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.SubDict.values() returns a MultiMappingView subclass."""
    sub1 = sample_foam_file['sub1']
    values_view = sub1.values()
    
    # Check type hierarchy
    assert isinstance(values_view, multicollections.abc.MultiMappingView)
    assert isinstance(values_view, multicollections.abc.ValuesView)
    
    # Check functionality
    assert hasattr(values_view, '__iter__')
    assert hasattr(values_view, '__len__')
    assert hasattr(values_view, '__contains__')
    
    # Test actual behavior
    values_list = list(values_view)
    assert 'value1' in values_list
    assert 'value2' in values_list
    assert len(values_view) == 2
    assert 'value1' in values_view


def test_subdict_items_returns_multimapping_view(sample_foam_file):
    """Test that FoamFile.SubDict.items() returns a MultiMappingView subclass."""
    sub1 = sample_foam_file['sub1']
    items_view = sub1.items()
    
    # Check type hierarchy
    assert isinstance(items_view, multicollections.abc.MultiMappingView)
    assert isinstance(items_view, multicollections.abc.ItemsView)
    
    # Check functionality
    assert hasattr(items_view, '__iter__')
    assert hasattr(items_view, '__len__')
    assert hasattr(items_view, '__contains__')
    
    # Test actual behavior
    items_list = list(items_view)
    expected_items = [('key1', 'value1'), ('key2', 'value2')]
    for item in expected_items:
        assert item in items_list
    assert len(items_view) == 2
    assert ('key1', 'value1') in items_view


def test_view_compatibility_with_existing_code(sample_foam_file):
    """Test that the view objects work with existing code that expects collections."""
    # Test that we can iterate over views like we used to with lists
    keys = sample_foam_file.keys()
    values = sample_foam_file.values()
    items = sample_foam_file.items()
    
    # These operations should work just like they did with lists
    assert len(keys) > 0
    assert len(values) > 0
    assert len(items) > 0
    
    # Test list conversion
    keys_as_list = list(keys)
    values_as_list = list(values)
    items_as_list = list(items)
    
    assert isinstance(keys_as_list, list)
    assert isinstance(values_as_list, list)
    assert isinstance(items_as_list, list)
    
    # Test that we can use them in comprehensions
    uppercase_keys = [k.upper() for k in keys if k is not None]
    assert len(uppercase_keys) > 0
    
    # Test with SubDict as well
    sub1 = sample_foam_file['sub1']
    sub_keys = sub1.keys()
    sub_values = sub1.values()
    sub_items = sub1.items()
    
    assert len(sub_keys) == 2
    assert len(sub_values) == 2
    assert len(sub_items) == 2
    
    # Test list conversion for SubDict views
    assert isinstance(list(sub_keys), list)
    assert isinstance(list(sub_values), list)
    assert isinstance(list(sub_items), list)


def test_views_with_parameters(sample_foam_file):
    """Test that view classes work with include_header parameter."""
    # Test with include_header=True
    keys_with_header = sample_foam_file.keys(include_header=True)
    keys_without_header = sample_foam_file.keys(include_header=False)
    
    assert isinstance(keys_with_header, multicollections.abc.KeysView)
    assert isinstance(keys_without_header, multicollections.abc.KeysView)
    
    # The header version should have one more item (FoamFile)
    assert len(keys_with_header) == len(keys_without_header) + 1
    
    # Test for other view types too
    values_with_header = sample_foam_file.values(include_header=True)
    items_with_header = sample_foam_file.items(include_header=True)
    
    assert isinstance(values_with_header, multicollections.abc.ValuesView)
    assert isinstance(items_with_header, multicollections.abc.ItemsView)