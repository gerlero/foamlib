"""Tests for foamlib.postprocessing.table_reader module - additional coverage."""

from pathlib import Path

import pytest
from foamlib.postprocessing.table_reader import ReaderNotRegisteredError, TableReader


def test_reader_not_registered_error(tmp_path: Path) -> None:
    """Test that reading a file with an unregistered extension raises an error."""
    # Create a temporary file with an unregistered extension
    tmp_file = tmp_path / "test.unknown"
    tmp_file.write_bytes(b"some data")

    reader = TableReader()

    with pytest.raises(
        ReaderNotRegisteredError,
        match=r"No reader registered for extension\: \'\.unknown\'",
    ):
        reader.read(tmp_file)


def test_tablereader_init() -> None:
    """Test TableReader initialization."""
    reader = TableReader()
    assert reader is not None
    assert hasattr(reader, "_registry")
