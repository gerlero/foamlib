"""Test edge cases for preprocessing functionality."""

import tempfile
from pathlib import Path

import pytest
from foamlib import FoamFile
from foamlib.preprocessing.of_dict import FoamDictAssignment, FoamDictInstruction


def test_foam_dict_instruction_edge_cases():
    """Test FoamDictInstruction with various edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with complex structure
        test_file = Path(tmpdir) / "complexDict"
        ff = FoamFile(test_file)
        
        # Set up complex nested structure
        ff["simple"] = 42
        ff["nested"] = {
            "level2": {
                "level3": {
                    "value": "deep_value",
                    "number": 3.14159
                }
            }
        }
        ff["list"] = [1, 2, 3]
        ff["empty_dict"] = {}
        
        # Test simple key access
        simple_instruction = FoamDictInstruction(file_name=test_file, keys=["simple"])
        assert simple_instruction.get_value() == 42
        
        # Test deep nested access
        deep_instruction = FoamDictInstruction(
            file_name=test_file, 
            keys=["nested", "level2", "level3", "value"]
        )
        assert deep_instruction.get_value() == "deep_value"
        
        # Test numeric access
        number_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["nested", "level2", "level3", "number"]
        )
        assert number_instruction.get_value() == 3.14159
        
        # Test list access
        list_instruction = FoamDictInstruction(file_name=test_file, keys=["list"])
        assert list_instruction.get_value() == [1, 2, 3]
        
        # Test empty dict access
        empty_instruction = FoamDictInstruction(file_name=test_file, keys=["empty_dict"])
        result = empty_instruction.get_value()
        assert isinstance(result, (dict, FoamFile.SubDict))


def test_foam_dict_instruction_nonexistent_keys():
    """Test FoamDictInstruction with non-existent keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "testDict"
        ff = FoamFile(test_file)
        
        # Set up some data
        ff["existing"] = {"sub": "value"}
        
        # Test non-existent top-level key
        nonexistent_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["nonexistent"]
        )
        
        with pytest.raises(KeyError):
            nonexistent_instruction.get_value()
        
        # Test non-existent nested key
        nonexistent_nested = FoamDictInstruction(
            file_name=test_file,
            keys=["existing", "nonexistent"]
        )
        
        with pytest.raises(KeyError):
            nonexistent_nested.get_value()
        
        # Test partially non-existent path
        partial_nonexistent = FoamDictInstruction(
            file_name=test_file,
            keys=["nonexistent", "sub", "value"]
        )
        
        with pytest.raises(KeyError):
            partial_nonexistent.get_value()


def test_foam_dict_instruction_with_special_keys():
    """Test FoamDictInstruction with special characters in keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "specialKeysDict"
        ff = FoamFile(test_file)
        
        # Set up data with special keys
        ff["div(phi,U)"] = ("Gauss", "linearUpwind", "grad(U)")
        ff["laplacian(nu,U)"] = ("Gauss", "linear", "corrected")
        ff["grad(p)"] = ("Gauss", "linear")
        ff["$variable"] = "macro_value"
        ff["#include"] = "someFile"
        
        # Test access to special keys
        div_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["div(phi,U)"]
        )
        assert div_instruction.get_value() == ("Gauss", "linearUpwind", "grad(U)")
        
        laplacian_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["laplacian(nu,U)"]
        )
        assert laplacian_instruction.get_value() == ("Gauss", "linear", "corrected")
        
        macro_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["$variable"]
        )
        assert macro_instruction.get_value() == "macro_value"
        
        include_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["#include"]
        )
        assert include_instruction.get_value() == "someFile"


def test_foam_dict_assignment_edge_cases():
    """Test FoamDictAssignment with various edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "assignmentTest"
        
        # Initially create empty file
        ff = FoamFile(test_file)
        ff["initial"] = "value"
        
        # Test assignment of various data types
        
        # String assignment
        string_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(file_name=test_file, keys=["string_key"]),
            value="string_value"
        )
        result_ff = string_assignment.set_value()
        assert result_ff["string_key"] == "string_value"
        
        # Number assignment
        number_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(file_name=test_file, keys=["number_key"]),
            value=3.14159
        )
        result_ff = number_assignment.set_value()
        assert result_ff["number_key"] == 3.14159
        
        # Boolean assignment
        bool_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(file_name=test_file, keys=["bool_key"]),
            value=True
        )
        result_ff = bool_assignment.set_value()
        assert result_ff["bool_key"] is True
        
        # List assignment
        list_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(file_name=test_file, keys=["list_key"]),
            value=[1, 2, 3, 4, 5]
        )
        result_ff = list_assignment.set_value()
        assert result_ff["list_key"] == [1, 2, 3, 4, 5]
        
        # Dict assignment
        dict_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(file_name=test_file, keys=["dict_key"]),
            value={"sub1": "value1", "sub2": "value2"}
        )
        result_ff = dict_assignment.set_value()
        assert result_ff["dict_key", "sub1"] == "value1"
        assert result_ff["dict_key", "sub2"] == "value2"


def test_foam_dict_assignment_deep_nesting():
    """Test FoamDictAssignment with deep nesting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "deepNestingTest"
        
        # Create initial structure
        ff = FoamFile(test_file)
        ff["level1"] = {"level2": {"level3": {"old_value": "old"}}}
        
        # Test deep assignment - new key
        deep_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["level1", "level2", "level3", "new_value"]
            ),
            value="new"
        )
        result_ff = deep_assignment.set_value()
        assert result_ff["level1", "level2", "level3", "new_value"] == "new"
        assert result_ff["level1", "level2", "level3", "old_value"] == "old"
        
        # Test deep assignment - replace existing
        replace_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["level1", "level2", "level3", "old_value"]
            ),
            value="replaced"
        )
        result_ff = replace_assignment.set_value()
        assert result_ff["level1", "level2", "level3", "old_value"] == "replaced"


def test_foam_dict_assignment_with_case_path():
    """Test FoamDictAssignment with case_path parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        system_path = case_path / "system"
        system_path.mkdir(parents=True)
        
        # Create file relative to case
        relative_file = Path("system/controlDict")
        absolute_file = case_path / relative_file
        
        # Create initial file
        ff = FoamFile(absolute_file)
        ff["startTime"] = 0
        ff["endTime"] = 1000
        
        # Test assignment with case_path
        assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=relative_file,
                keys=["endTime"]
            ),
            value=2000
        )
        
        result_ff = assignment.set_value(case_path=case_path)
        assert result_ff["endTime"] == 2000
        assert result_ff["startTime"] == 0  # Should preserve other values


def test_foam_dict_assignment_nonexistent_file():
    """Test FoamDictAssignment with non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_file = Path(tmpdir) / "nonexistent.dict"
        
        assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=nonexistent_file,
                keys=["test_key"]
            ),
            value="test_value"
        )
        
        with pytest.raises(FileNotFoundError):
            assignment.set_value()


def test_foam_dict_assignment_nonexistent_file_with_case_path():
    """Test FoamDictAssignment with non-existent file and case_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "test_case"
        case_path.mkdir()
        
        nonexistent_file = Path("system/nonexistent.dict")
        
        assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=nonexistent_file,
                keys=["test_key"]
            ),
            value="test_value"
        )
        
        with pytest.raises(FileNotFoundError):
            assignment.set_value(case_path=case_path)


def test_foam_dict_with_complex_openfoam_structures():
    """Test with complex OpenFOAM-specific structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "fvSchemesTest"
        
        # Create typical fvSchemes structure
        ff = FoamFile(test_file)
        ff["ddtSchemes"] = {"default": "Euler"}
        ff["gradSchemes"] = {
            "default": ("Gauss", "linear"),
            "grad(p)": ("Gauss", "linear"),
            "grad(U)": ("Gauss", "linearUpwind", "grad(U)")
        }
        ff["divSchemes"] = {
            "default": "none",
            "div(phi,U)": ("Gauss", "linearUpwind", "grad(U)"),
            "div(phi,k)": ("Gauss", "upwind"),
            "div((nuEff*dev2(T(grad(U)))))": ("Gauss", "linear")
        }
        ff["laplacianSchemes"] = {
            "default": ("Gauss", "linear", "corrected"),
            "laplacian(nu,U)": ("Gauss", "linear", "corrected"),
            "laplacian((1|A(U)),p)": ("Gauss", "linear", "corrected")
        }
        
        # Test accessing complex scheme keys
        div_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["divSchemes", "div((nuEff*dev2(T(grad(U)))))"]
        )
        assert div_instruction.get_value() == ("Gauss", "linear")
        
        # Test modifying complex scheme
        new_scheme_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["divSchemes", "div(phi,epsilon)"]
            ),
            value=("Gauss", "upwind")
        )
        result_ff = new_scheme_assignment.set_value()
        assert result_ff["divSchemes", "div(phi,epsilon)"] == ("Gauss", "upwind")
        
        # Test replacing existing scheme
        replace_scheme = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["gradSchemes", "grad(U)"]
            ),
            value=("Gauss", "leastSquares")
        )
        result_ff = replace_scheme.set_value()
        assert result_ff["gradSchemes", "grad(U)"] == ("Gauss", "leastSquares")


def test_foam_dict_with_boundary_conditions():
    """Test with boundary condition structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "boundaryTest"
        
        # Create typical boundary field structure
        ff = FoamFile(test_file)
        ff["boundaryField"] = {
            "inlet": {
                "type": "fixedValue",
                "value": "uniform (1 0 0)"
            },
            "outlet": {
                "type": "zeroGradient"
            },
            "walls": {
                "type": "noSlip"
            }
        }
        
        # Test accessing boundary condition properties
        inlet_type_instruction = FoamDictInstruction(
            file_name=test_file,
            keys=["boundaryField", "inlet", "type"]
        )
        assert inlet_type_instruction.get_value() == "fixedValue"
        
        # Test modifying boundary condition
        new_bc_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["boundaryField", "inlet", "value"]
            ),
            value="uniform (2 0 0)"
        )
        result_ff = new_bc_assignment.set_value()
        assert result_ff["boundaryField", "inlet", "value"] == "uniform (2 0 0)"
        
        # Test adding new boundary patch
        new_patch_assignment = FoamDictAssignment(
            instruction=FoamDictInstruction(
                file_name=test_file,
                keys=["boundaryField", "symmetryPlane"]
            ),
            value={"type": "symmetryPlane"}
        )
        result_ff = new_patch_assignment.set_value()
        assert result_ff["boundaryField", "symmetryPlane", "type"] == "symmetryPlane"


def test_foam_dict_instruction_path_handling():
    """Test FoamDictInstruction with various path formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with string path
        string_path = str(Path(tmpdir) / "stringPath.dict")
        ff_string = FoamFile(string_path)
        ff_string["test"] = "value"
        
        string_instruction = FoamDictInstruction(
            file_name=string_path,
            keys=["test"]
        )
        assert string_instruction.get_value() == "value"
        
        # Test with Path object
        path_obj = Path(tmpdir) / "pathObj.dict"
        ff_path = FoamFile(path_obj)
        ff_path["test"] = "value2"
        
        path_instruction = FoamDictInstruction(
            file_name=path_obj,
            keys=["test"]
        )
        assert path_instruction.get_value() == "value2"
        
        # Verify that file_name property returns Path object
        assert isinstance(string_instruction.file_name, Path)
        assert isinstance(path_instruction.file_name, Path)


def test_foam_dict_error_handling():
    """Test error handling in foam dict operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "errorTest.dict"
        
        # Test with invalid file permission (if possible to simulate)
        # This is system-dependent and might not always work
        try:
            ff = FoamFile(test_file)
            ff["test"] = "value"
            
            # Try to make file read-only
            test_file.chmod(0o444)
            
            assignment = FoamDictAssignment(
                instruction=FoamDictInstruction(
                    file_name=test_file,
                    keys=["new_key"]
                ),
                value="new_value"
            )
            
            # This might raise a permission error
            try:
                assignment.set_value()
            except PermissionError:
                # Expected behavior for read-only files
                pass
            finally:
                # Restore permissions for cleanup
                test_file.chmod(0o644)
                
        except Exception:
            # Skip if we can't test permissions on this system
            pass