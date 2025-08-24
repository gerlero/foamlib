"""Test edge cases and error handling for FoamCase."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from foamlib import FoamCase, FoamFieldFile


def test_foamcase_nonexistent_path():
    """Test FoamCase behavior with non-existent paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = Path(tmpdir) / "nonexistent"
        case = FoamCase(nonexistent)
        
        # Path should be set even if it doesn't exist
        assert case.path == nonexistent.absolute()
        assert str(case) == str(nonexistent.absolute())
        assert case.__fspath__() == str(nonexistent.absolute())


def test_foamcase_empty_directory():
    """Test FoamCase with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case = FoamCase(tmpdir)
        
        # Should handle empty directory gracefully
        assert len(case) == 0
        assert list(case) == []
        
        # Should raise appropriate errors for missing files
        with pytest.raises(FileNotFoundError):
            _ = case.control_dict


def test_foamcase_time_directory_edge_cases():
    """Test TimeDirectory edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        case_path.mkdir()
        
        # Create time directories with edge case names
        time_dirs = ["0", "0.5", "1e-6", "100.0", "constant"]
        for time_dir in time_dirs:
            (case_path / time_dir).mkdir()
            
        case = FoamCase(case_path)
        
        # Test numeric time directories
        if (case_path / "0").exists():
            time_0 = case[0]
            assert time_0.time == 0.0
            assert time_0.name == "0"
            
        if (case_path / "0.5").exists():
            time_05 = case[0.5] 
            assert time_05.time == 0.5
            assert time_05.name == "0.5"


def test_time_directory_field_access():
    """Test TimeDirectory field access with various scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        time_path = case_path / "0"
        time_path.mkdir(parents=True)
        
        # Create various field files
        field_files = ["U", "p", "T.gz", "k"]
        for field in field_files:
            (time_path / field).touch()
            
        case = FoamCase(case_path)
        time_dir = case[0]
        
        # Test normal field access
        assert "U" in time_dir
        assert "p" in time_dir 
        assert "k" in time_dir
        
        # Test compressed field access
        assert "T" in time_dir  # Should find T.gz
        
        # Test field file object creation
        U_field = time_dir["U"]
        assert isinstance(U_field, FoamFieldFile)
        assert U_field.path == time_path / "U"
        
        # Test compressed field file object
        T_field = time_dir["T"]
        assert isinstance(T_field, FoamFieldFile)
        assert T_field.path == time_path / "T.gz"
        
        # Test iteration over fields
        fields = list(time_dir)
        assert len(fields) == len(field_files)
        
        # Test field deletion
        del time_dir["k"]
        assert not (time_path / "k").exists()
        assert "k" not in time_dir


def test_time_directory_compressed_vs_uncompressed():
    """Test TimeDirectory handling of compressed vs uncompressed files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case" 
        time_path = case_path / "0"
        time_path.mkdir(parents=True)
        
        # Create both compressed and uncompressed versions
        (time_path / "U").touch()
        (time_path / "U.gz").touch()
        
        case = FoamCase(case_path)
        time_dir = case[0]
        
        # Should prefer uncompressed version
        U_field = time_dir["U"]
        assert U_field.path == time_path / "U"
        
        # Remove uncompressed version
        (time_path / "U").unlink()
        
        # Should now use compressed version
        U_field = time_dir["U"]
        assert U_field.path == time_path / "U.gz"


def test_foamcase_file_access():
    """Test FoamCase file access methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        system_path = case_path / "system"
        constant_path = case_path / "constant"
        system_path.mkdir(parents=True)
        constant_path.mkdir(parents=True)
        
        # Create some standard files
        files_to_create = [
            "system/controlDict",
            "system/fvSchemes", 
            "system/fvSolution",
            "system/decomposeParDict",
            "system/blockMeshDict",
            "constant/transportProperties",
            "constant/turbulenceProperties"
        ]
        
        for file_path in files_to_create:
            (case_path / file_path).touch()
            
        case = FoamCase(case_path)
        
        # Test property access to standard files
        assert case.control_dict.path.name == "controlDict"
        assert case.fv_schemes.path.name == "fvSchemes"
        assert case.fv_solution.path.name == "fvSolution"
        assert case.decompose_par_dict.path.name == "decomposeParDict"
        assert case.block_mesh_dict.path.name == "blockMeshDict"
        assert case.transport_properties.path.name == "transportProperties"
        assert case.turbulence_properties.path.name == "turbulenceProperties"


def test_foamcase_indexing_edge_cases():
    """Test FoamCase indexing with various edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        case_path.mkdir()
        
        # Create time directories with unusual names
        time_dirs = ["0", "1.5", "2e-3", "1000", "latestTime"]
        for time_dir in time_dirs[:-1]:  # Skip "latestTime" for now
            (case_path / time_dir).mkdir()
            
        case = FoamCase(case_path)
        
        # Test numeric indexing
        assert case[0].name == "0"
        assert case[-1].name == "1000"  # Last time
        
        # Test string indexing  
        assert case["0"].name == "0"
        assert case["1.5"].name == "1.5"
        
        # Test float indexing
        assert case[0.0].name == "0"
        assert case[1.5].name == "1.5"


def test_foamcase_boolean_evaluation():
    """Test FoamCase boolean evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty case
        empty_case = FoamCase(tmpdir)
        assert not empty_case  # Should be falsy when empty
        
        # Case with time directories
        case_path = Path(tmpdir) / "with_times"
        case_path.mkdir()
        (case_path / "0").mkdir()
        
        non_empty_case = FoamCase(case_path)
        assert non_empty_case  # Should be truthy when has time dirs


def test_foamcase_representation():
    """Test FoamCase string representations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case = FoamCase(tmpdir)
        
        repr_str = repr(case)
        assert "FoamCase" in repr_str
        assert tmpdir in repr_str
        
        str_repr = str(case)
        assert tmpdir in str_repr


def test_time_directory_representation():
    """Test TimeDirectory string representation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        time_path = case_path / "0"
        time_path.mkdir(parents=True)
        
        case = FoamCase(case_path)
        time_dir = case[0]
        
        repr_str = repr(time_dir)
        assert "TimeDirectory" in repr_str
        assert str(time_path) in repr_str


def test_foamcase_with_complex_time_structure():
    """Test FoamCase with complex time directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        case_path.mkdir()
        
        # Create time directories with various formats
        times = [
            "0",
            "1", 
            "1.5",
            "2.0",
            "10",
            "1e-06",
            "999.999",
            "constant"  # Should be ignored for time indexing
        ]
        
        for time_name in times:
            (case_path / time_name).mkdir()
            
        case = FoamCase(case_path)
        
        # Test that times are sorted correctly
        time_values = [t.time for t in case if t.name != "constant"]
        assert time_values == sorted(time_values)
        
        # Test access to different time formats
        assert case[1e-6].name == "1e-06"
        assert case[999.999].name == "999.999"


def test_foamcase_file_method():
    """Test FoamCase.file() method with various paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        case_path.mkdir()
        
        case = FoamCase(case_path)
        
        # Test with relative path
        file_obj = case.file("system/controlDict")
        assert file_obj.path == case_path / "system" / "controlDict"
        
        # Test with Path object
        file_obj = case.file(Path("constant/transportProperties"))
        assert file_obj.path == case_path / "constant" / "transportProperties"


def test_time_directory_length_and_membership():
    """Test TimeDirectory length calculation and membership."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        time_path = case_path / "0"
        time_path.mkdir(parents=True)
        
        # Create mix of field files and other files
        files = ["U", "p", "T.gz", "k", "README.txt", ".hidden"]
        for file_name in files:
            (time_path / file_name).touch()
            
        case = FoamCase(case_path)
        time_dir = case[0]
        
        # Test length - should count all valid files
        field_files = list(time_dir)
        assert len(time_dir) == len(field_files)
        
        # Test membership with FoamFieldFile objects
        U_field = time_dir["U"]
        assert U_field in time_dir
        
        # Create field file from different time directory
        other_time_path = case_path / "1"
        other_time_path.mkdir()
        (other_time_path / "U").touch()
        other_U = FoamFieldFile(other_time_path / "U")
        assert other_U not in time_dir


def test_foamcase_clone_context_manager():
    """Test FoamCase.clone() as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "original_case"
        case_path.mkdir()
        (case_path / "0").mkdir()
        (case_path / "0" / "U").touch()
        
        case = FoamCase(case_path)
        
        # Test clone context manager
        with case.clone() as clone:
            assert clone.path != case.path
            assert clone.path.exists()
            assert (clone.path / "0" / "U").exists()
            
            clone_path = clone.path
            
        # Clone should be cleaned up after context
        assert not clone_path.exists()


def test_foamcase_clone_with_ignore():
    """Test FoamCase.clone() with ignore patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "original_case"
        case_path.mkdir()
        
        # Create various files and directories
        (case_path / "0").mkdir()
        (case_path / "0" / "U").touch()
        (case_path / "processor0").mkdir()
        (case_path / "processor0" / "0").mkdir()
        (case_path / "processor0" / "0" / "U").touch()
        (case_path / "postProcessing").mkdir()
        (case_path / "postProcessing" / "data.txt").touch()
        
        case = FoamCase(case_path)
        
        # Clone ignoring processor directories
        def ignore_processors(path, names):
            return [name for name in names if name.startswith("processor")]
            
        with case.clone(ignore=ignore_processors) as clone:
            assert (clone.path / "0" / "U").exists()
            assert not (clone.path / "processor0").exists()
            assert (clone.path / "postProcessing").exists()


def test_foamcase_properties_with_missing_files():
    """Test FoamCase property access when files don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case = FoamCase(tmpdir)
        
        # Properties should still return FoamFile objects even if files don't exist
        control_dict = case.control_dict
        assert control_dict.path.name == "controlDict"
        
        fv_schemes = case.fv_schemes
        assert fv_schemes.path.name == "fvSchemes"
        
        # But accessing data should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            _ = control_dict["startTime"]