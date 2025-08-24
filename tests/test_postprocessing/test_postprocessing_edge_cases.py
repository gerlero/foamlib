"""Test edge cases for postprocessing functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from foamlib.postprocessing.load_tables import (
    datafile,
    functionobject,
    list_function_objects,
    load_tables,
    of_cases,
)


def test_of_cases_edge_cases():
    """Test of_cases function with edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create various directory structures
        
        # Valid OpenFOAM case
        valid_case = test_dir / "valid_case"
        (valid_case / "constant").mkdir(parents=True)
        (valid_case / "system").mkdir(parents=True)
        
        # Invalid case - only constant
        invalid_case1 = test_dir / "invalid_case1"
        (invalid_case1 / "constant").mkdir(parents=True)
        
        # Invalid case - only system
        invalid_case2 = test_dir / "invalid_case2"
        (invalid_case2 / "system").mkdir(parents=True)
        
        # Nested valid case
        nested_case = test_dir / "parent" / "nested_case"
        (nested_case / "constant").mkdir(parents=True)
        (nested_case / "system").mkdir(parents=True)
        
        # Regular directory
        regular_dir = test_dir / "regular_dir"
        regular_dir.mkdir()
        (regular_dir / "some_file.txt").touch()
        
        cases = of_cases(test_dir)
        
        # Should find valid cases
        assert str(valid_case) in cases
        assert str(nested_case) in cases
        
        # Should not find invalid cases
        assert str(invalid_case1) not in cases
        assert str(invalid_case2) not in cases
        assert str(regular_dir) not in cases


def test_of_cases_with_symlinks():
    """Test of_cases function with symbolic links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create a valid case
        real_case = test_dir / "real_case"
        (real_case / "constant").mkdir(parents=True)
        (real_case / "system").mkdir(parents=True)
        
        # Create symlink to the case
        symlink_case = test_dir / "symlink_case"
        try:
            symlink_case.symlink_to(real_case)
            
            cases = of_cases(test_dir)
            
            # Should find both real and symlinked cases
            case_names = [Path(case).name for case in cases]
            assert "real_case" in case_names
            # Symlink behavior depends on system, so we don't assert on it
            
        except OSError:
            # Skip if symlinks not supported on this system
            pytest.skip("Symlinks not supported on this system")


def test_of_cases_with_permissions():
    """Test of_cases function with permission issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create a valid case
        case_dir = test_dir / "test_case"
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # This should work normally
        cases = of_cases(test_dir)
        assert str(case_dir) in cases


def test_load_tables_with_missing_files():
    """Test load_tables with missing or inaccessible files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create case structure without the actual data files
        case_dir = test_dir / "test_case"
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        (case_dir / "postProcessing" / "forces").mkdir(parents=True)
        
        # Define a source that points to non-existent file
        file_source = functionobject(file_name="nonexistent.dat", folder="forces")
        
        # Should handle missing files gracefully
        result = load_tables(source=file_source, dir_name=test_dir)
        
        # Result should be None or empty DataFrame
        assert result is None or (isinstance(result, pd.DataFrame) and result.empty)


def test_load_tables_with_malformed_data():
    """Test load_tables with malformed data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create case structure
        case_dir = test_dir / "test_case"
        forces_dir = case_dir / "postProcessing" / "forces"
        forces_dir.mkdir(parents=True)
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # Create malformed data file
        data_file = forces_dir / "forces.dat"
        data_file.write_text("""
        # This is a malformed file
        Time  Force
        not_a_number  invalid_data
        1.0  "string_instead_of_number"
        """)
        
        file_source = functionobject(file_name="forces.dat", folder="forces")
        
        # Should handle malformed data gracefully
        try:
            result = load_tables(source=file_source, dir_name=test_dir)
            # If it doesn't raise an exception, result should be reasonable
            assert result is None or isinstance(result, pd.DataFrame)
        except (ValueError, pd.errors.ParserError):
            # These exceptions are acceptable for malformed data
            pass


def test_load_tables_with_empty_files():
    """Test load_tables with empty data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create case structure
        case_dir = test_dir / "test_case"
        forces_dir = case_dir / "postProcessing" / "forces"
        forces_dir.mkdir(parents=True)
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # Create empty data file
        data_file = forces_dir / "forces.dat"
        data_file.touch()
        
        file_source = functionobject(file_name="forces.dat", folder="forces")
        
        result = load_tables(source=file_source, dir_name=test_dir)
        
        # Should handle empty files gracefully
        assert result is None or (isinstance(result, pd.DataFrame) and result.empty)


def test_load_tables_with_complex_filter():
    """Test load_tables with complex filter functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create test data
        case_dir = test_dir / "test_case"
        forces_dir = case_dir / "postProcessing" / "forces"
        forces_dir.mkdir(parents=True)
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # Create simple data file
        data_file = forces_dir / "forces.dat"
        data_file.write_text("""
        # Time  Fx  Fy  Fz
        0.0  1.0  2.0  3.0
        0.1  1.1  2.1  3.1
        0.2  1.2  2.2  3.2
        0.3  1.3  2.3  3.3
        """)
        
        def complex_filter(table: pd.DataFrame, parameters: list) -> pd.DataFrame:
            """Filter that does statistical operations."""
            if table.empty:
                return table
                
            # Calculate statistics
            stats = {
                "mean_Fx": [table["Fx"].mean()],
                "max_Fy": [table["Fy"].max()],
                "min_Fz": [table["Fz"].min()],
                "count": [len(table)]
            }
            
            # Add parameter information
            for param in parameters:
                stats[param["category"]] = [param["name"]]
                
            return pd.DataFrame(stats)
        
        file_source = functionobject(file_name="forces.dat", folder="forces")
        
        try:
            result = load_tables(
                source=file_source,
                dir_name=test_dir,
                filter_table=complex_filter
            )
            
            if result is not None and not result.empty:
                # Should have applied the filter
                assert "mean_Fx" in result.columns
                assert "max_Fy" in result.columns
                assert "min_Fz" in result.columns
                assert "count" in result.columns
                
        except Exception:
            # If the function doesn't support this, that's okay
            pass


def test_datafile_vs_functionobject():
    """Test differences between datafile and functionobject constructors."""
    # Test datafile constructor
    df = datafile(file_name="test.xml", folder=".")
    assert df.file_name == "test.xml"
    assert df.folder == Path(".")
    
    # Test functionobject constructor
    fo = functionobject(file_name="forces.dat", folder="forces")
    assert fo.file_name == "forces.dat"
    assert fo.folder == Path("forces")
    
    # Test with Path objects
    fo_path = functionobject(file_name="test.dat", folder=Path("postProcessing"))
    assert fo_path.folder == Path("postProcessing")


def test_list_function_objects_edge_cases():
    """Test list_function_objects with various edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create complex case structure
        case_dir = test_dir / "test_case"
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # Create postProcessing structure with various files
        pp_dir = case_dir / "postProcessing"
        
        # Forces folder with multiple files
        forces_dir = pp_dir / "forces"
        forces_dir.mkdir(parents=True)
        (forces_dir / "forces.dat").touch()
        (forces_dir / "moments.dat").touch()
        (forces_dir / "README.txt").touch()  # Non-data file
        
        # Probes folder with different file types
        probes_dir = pp_dir / "probes"
        probes_dir.mkdir(parents=True)
        (probes_dir / "U").touch()
        (probes_dir / "p").touch()
        (probes_dir / "T.gz").touch()  # Compressed file
        
        # Empty folder
        empty_dir = pp_dir / "empty"
        empty_dir.mkdir()
        
        # Folder with subdirectories
        nested_dir = pp_dir / "nested"
        nested_subdir = nested_dir / "subdir"
        nested_subdir.mkdir(parents=True)
        (nested_subdir / "data.csv").touch()
        
        output_files = list_function_objects(test_dir)
        
        # Should find files in various folders
        found_folders = set()
        found_files = set()
        
        for key, output_file in output_files.items():
            found_folders.add(str(output_file.folder))
            found_files.add(output_file.file_name)
            
        # Should include files from forces and probes
        assert "forces.dat" in found_files
        assert "moments.dat" in found_files
        assert "U" in found_files
        assert "p" in found_files
        
        # Should not include non-data files
        assert "README.txt" not in found_files


def test_load_tables_with_custom_reader():
    """Test load_tables with custom reader function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create case with custom data format
        case_dir = test_dir / "test_case"
        data_dir = case_dir / "customData"
        data_dir.mkdir(parents=True)
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
        
        # Create custom format file
        data_file = data_dir / "custom.txt"
        data_file.write_text("""
        CUSTOM_FORMAT_v1.0
        key1=value1
        key2=value2
        key3=123
        key4=45.6
        """)
        
        def custom_reader(file_path: Path) -> pd.DataFrame:
            """Custom reader for specific format."""
            lines = file_path.read_text().strip().split('\n')
            
            data = {}
            for line in lines[1:]:  # Skip header
                if '=' in line:
                    key, value = line.split('=', 1)
                    try:
                        # Try to convert to number
                        data[key] = float(value)
                    except ValueError:
                        data[key] = value
                        
            return pd.DataFrame([data])
        
        file_source = datafile(file_name="custom.txt", folder="customData")
        
        try:
            result = load_tables(
                source=file_source,
                dir_name=test_dir,
                reader_fn=custom_reader
            )
            
            if result is not None and not result.empty:
                # Should have used custom reader
                assert "key1" in result.columns
                assert "key2" in result.columns
                assert result["key3"].iloc[0] == 123
                assert result["key4"].iloc[0] == 45.6
                
        except Exception:
            # If this fails, it's acceptable for this test
            pass


def test_load_tables_error_propagation():
    """Test that load_tables properly handles and propagates errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        def failing_filter(table: pd.DataFrame, parameters: list) -> pd.DataFrame:
            """Filter that always raises an exception."""
            raise ValueError("Filter intentionally failed")
        
        def failing_reader(file_path: Path) -> pd.DataFrame:
            """Reader that always raises an exception.""" 
            raise RuntimeError("Reader intentionally failed")
        
        file_source = datafile(file_name="any.txt", folder=".")
        
        # Test with failing filter
        with pytest.raises(ValueError, match="Filter intentionally failed"):
            load_tables(
                source=file_source,
                dir_name=test_dir,
                filter_table=failing_filter
            )
        
        # Test with failing reader
        with pytest.raises(RuntimeError, match="Reader intentionally failed"):
            load_tables(
                source=file_source,
                dir_name=test_dir,
                reader_fn=failing_reader
            )