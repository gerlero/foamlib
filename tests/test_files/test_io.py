"""Tests for foamlib._files._io module - additional coverage."""

import gzip
import tempfile
from pathlib import Path

from foamlib._files._io import FoamFileIO


def test_foam_file_io_gzip_read() -> None:
    """Test FoamFileIO reading a gzipped file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.gz"

        # Create a gzipped file
        content = b"FoamFile\n{\n    version     2.0;\n}\n"
        test_file.write_bytes(gzip.compress(content))

        # Read it with FoamFileIO
        io = FoamFileIO(test_file)
        parsed = io._get_parsed()

        # Verify the content was decompressed
        assert parsed.contents == bytearray(content)


def test_foam_file_io_gzip_write() -> None:
    """Test FoamFileIO writing a gzipped file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.gz"

        # Create and modify a file
        io = FoamFileIO(test_file)
        with io:
            parsed = io._get_parsed(missing_ok=True)
            parsed.contents = bytearray(b"FoamFile\n{\n    version     2.0;\n}\n")
            parsed.modified = True

        # Verify it was written as gzipped
        compressed_content = test_file.read_bytes()
        decompressed_content = gzip.decompress(compressed_content)
        assert b"FoamFile" in decompressed_content
        assert b"version     2.0;" in decompressed_content


def test_foam_file_io_repr() -> None:
    """Test FoamFileIO.__repr__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_file"
        io = FoamFileIO(test_file)

        repr_str = repr(io)
        assert "FoamFileIO" in repr_str
        assert "test_file" in repr_str


def test_foam_file_io_context_no_changes() -> None:
    """Test FoamFileIO context manager when no changes are made."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_file"
        test_file.write_bytes(b"FoamFile\n{\n    version     2.0;\n}\n")

        io = FoamFileIO(test_file)
        original_content = test_file.read_bytes()

        with io:
            # Read but don't modify
            parsed = io._get_parsed()
            assert parsed.contents is not None

        # File should remain unchanged
        assert test_file.read_bytes() == original_content
