"""Tests for foamlib.postprocessing.table_reader module - additional coverage."""

import tempfile
from pathlib import Path

import pytest
from foamlib.postprocessing.table_reader import ReaderNotRegisteredError, TableReader


def test_reader_not_registered_error() -> None:
    """Test that reading a file with an unregistered extension raises an error."""
    # Create a temporary file with an unregistered extension
    with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(b"some data")
    
    reader = TableReader()
    
    try:
        with pytest.raises(ReaderNotRegisteredError, match="No reader registered for extension: '.unknown'"):
            reader.read(tmp_path)
    finally:
        tmp_path.unlink()


def test_tablereader_init() -> None:
    """Test TableReader initialization."""
    reader = TableReader()
    assert reader is not None
    assert hasattr(reader, '_registry')
