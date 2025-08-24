"""Test complex field operations and data structure handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from foamlib import FoamFieldFile, FoamFile


def test_field_file_complex_internal_fields():
    """Test FoamFieldFile with complex internal field structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "U"
        ff = FoamFieldFile(test_file)
        
        # Test various field shapes and types
        
        # Scalar field
        scalar_field = np.random.rand(100)
        ff.internal_field = scalar_field
        recovered_scalar = ff.internal_field
        assert np.allclose(scalar_field, recovered_scalar)
        assert ff.class_ == "volScalarField"
        
        # Vector field
        vector_field = np.random.rand(100, 3)
        ff.internal_field = vector_field
        recovered_vector = ff.internal_field
        assert np.allclose(vector_field, recovered_vector)
        assert ff.class_ == "volVectorField"
        
        # Tensor field (9 components)
        tensor_field = np.random.rand(50, 9)
        ff.internal_field = tensor_field
        recovered_tensor = ff.internal_field
        assert np.allclose(tensor_field, recovered_tensor)
        assert ff.class_ == "volTensorField"
        
        # Symmetric tensor field (6 components)
        symmtensor_field = np.random.rand(75, 6)
        ff.internal_field = symmtensor_field
        recovered_symmtensor = ff.internal_field
        assert np.allclose(symmtensor_field, recovered_symmtensor)
        assert ff.class_ == "volSymmTensorField"


def test_field_file_edge_case_shapes():
    """Test FoamFieldFile with edge case field shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "testField"
        ff = FoamFieldFile(test_file)
        
        # Single cell scalar
        single_scalar = np.array([42.0])
        ff.internal_field = single_scalar
        assert np.allclose(ff.internal_field, single_scalar)
        
        # Single cell vector
        single_vector = np.array([[1.0, 2.0, 3.0]])
        ff.internal_field = single_vector
        assert np.allclose(ff.internal_field, single_vector)
        
        # Large field
        large_field = np.random.rand(10000, 3)
        ff.internal_field = large_field
        assert np.allclose(ff.internal_field, large_field)
        
        # Field with zeros
        zero_field = np.zeros((100, 3))
        ff.internal_field = zero_field
        assert np.allclose(ff.internal_field, zero_field)
        
        # Field with extreme values
        extreme_field = np.array([[-1e10, 0, 1e10], [np.inf, -np.inf, 1e-15]])
        ff.internal_field = extreme_field
        recovered = ff.internal_field
        # Handle infinite values carefully
        finite_mask = np.isfinite(extreme_field)
        assert np.allclose(recovered[finite_mask], extreme_field[finite_mask])


def test_field_file_invalid_shapes():
    """Test FoamFieldFile with invalid field shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "invalidField"
        ff = FoamFieldFile(test_file)
        
        # Invalid tensor shape (not 3, 6, or 9 components)
        with pytest.raises(ValueError, match="Invalid field shape"):
            ff.internal_field = np.random.rand(10, 5)
        
        # Invalid dimension (3D array)
        with pytest.raises(ValueError, match="Invalid field shape"):
            ff.internal_field = np.random.rand(10, 3, 3)
        
        # Invalid dimension (4D array)
        with pytest.raises(ValueError, match="Invalid field shape"):
            ff.internal_field = np.random.rand(5, 5, 3, 3)


def test_field_file_boundary_field_complex():
    """Test complex boundary field operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "U"
        ff = FoamFieldFile(test_file)
        
        # Set up field structure
        ff.internal_field = np.random.rand(100, 3)
        
        # Create complex boundary field structure
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
            },
            "movingWall": {
                "type": "movingWallVelocity",
                "value": "uniform (0 1 0)"
            }
        }
        
        # Test boundary field access
        boundary_field = ff.boundary_field
        assert "inlet" in boundary_field
        assert "outlet" in boundary_field
        assert "walls" in boundary_field
        assert "movingWall" in boundary_field
        
        # Test individual boundary patch access
        inlet = boundary_field["inlet"]
        assert inlet.type == "fixedValue"
        assert inlet.value == "uniform (1 0 0)"
        
        outlet = boundary_field["outlet"]
        assert outlet.type == "zeroGradient"
        assert "value" not in outlet
        
        # Test boundary field modification
        inlet.value = [2, 0, 0]
        assert inlet.value == [2, 0, 0]
        
        # Test adding new boundary patch
        boundary_field["symmetryPlane"] = {
            "type": "symmetry"
        }
        assert "symmetryPlane" in boundary_field
        assert boundary_field["symmetryPlane"].type == "symmetry"


def test_field_file_dimensions_complex():
    """Test complex dimension handling in field files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "complexField"
        ff = FoamFieldFile(test_file)
        
        # Test various dimension combinations
        from foamlib._files._types import DimensionSet
        
        # Velocity dimensions
        velocity_dims = DimensionSet(length=1, time=-1)
        ff.dimensions = velocity_dims
        assert ff.dimensions == velocity_dims
        
        # Pressure dimensions
        pressure_dims = DimensionSet(mass=1, length=-1, time=-2)
        ff.dimensions = pressure_dims
        assert ff.dimensions == pressure_dims
        
        # Temperature dimensions
        temp_dims = DimensionSet(temperature=1)
        ff.dimensions = temp_dims
        assert ff.dimensions == temp_dims
        
        # Dimensionless
        dimensionless = DimensionSet()
        ff.dimensions = dimensionless
        assert ff.dimensions == dimensionless
        
        # Complex combination
        complex_dims = DimensionSet(mass=2, length=-3, time=1, temperature=-1, luminousIntensity=0.5)
        ff.dimensions = complex_dims
        assert ff.dimensions == complex_dims


def test_foamfile_complex_data_structures():
    """Test FoamFile with complex nested data structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "complexDict"
        ff = FoamFile(test_file)
        
        # Create deeply nested structure with various data types
        complex_structure = {
            "solver": {
                "type": "PISO",
                "nCorrectors": 2,
                "tolerance": 1e-6,
                "relTol": 0.1,
                "active": True
            },
            "relaxationFactors": {
                "fields": {
                    "p": 0.3,
                    "U": 0.7
                },
                "equations": {
                    "U": 0.9,
                    "k": 0.8,
                    "epsilon": 0.8
                }
            },
            "schemes": {
                "ddtSchemes": {
                    "default": "Euler"
                },
                "gradSchemes": {
                    "default": ("Gauss", "linear"),
                    "grad(p)": ("Gauss", "linear"),
                    "grad(U)": ("Gauss", "linearUpwind", "grad(U)")
                },
                "divSchemes": {
                    "default": "none",
                    "div(phi,U)": ("Gauss", "linearUpwind", "grad(U)"),
                    "div(phi,k)": ("Gauss", "upwind"),
                    "div((nuEff*dev2(T(grad(U)))))": ("Gauss", "linear")
                }
            },
            "boundary": {
                "inlet": {
                    "type": "patch",
                    "faces": [[0, 1, 2, 3], [4, 5, 6, 7]]
                },
                "outlet": {
                    "type": "patch", 
                    "faces": [[8, 9, 10, 11]]
                }
            },
            "initialConditions": {
                "U": {
                    "dimensions": [0, 1, -1, 0, 0, 0, 0],
                    "internalField": "uniform (0 0 0)",
                    "boundaryField": {
                        "inlet": {
                            "type": "fixedValue",
                            "value": "uniform (1 0 0)"
                        }
                    }
                }
            }
        }
        
        ff.update(complex_structure)
        
        # Test access to nested structures
        assert ff["solver", "type"] == "PISO"
        assert ff["solver", "nCorrectors"] == 2
        assert ff["solver", "tolerance"] == 1e-6
        assert ff["solver", "active"] is True
        
        assert ff["relaxationFactors", "fields", "p"] == 0.3
        assert ff["schemes", "gradSchemes", "grad(U)"] == ("Gauss", "linearUpwind", "grad(U)")
        assert ff["boundary", "inlet", "faces"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        
        # Test modification of nested structures
        ff["solver", "tolerance"] = 1e-8
        assert ff["solver", "tolerance"] == 1e-8
        
        # Test adding to nested structures
        ff["schemes", "divSchemes", "div(phi,epsilon)"] = ("Gauss", "upwind")
        assert ff["schemes", "divSchemes", "div(phi,epsilon)"] == ("Gauss", "upwind")


def test_foamfile_with_arrays_and_matrices():
    """Test FoamFile with numpy arrays and matrix operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "matrixDict"
        ff = FoamFile(test_file)
        
        # Test various array types
        
        # 1D array
        array_1d = np.array([1, 2, 3, 4, 5])
        ff["array1D"] = array_1d
        recovered_1d = ff["array1D"]
        assert np.array_equal(recovered_1d, array_1d)
        
        # 2D array
        array_2d = np.array([[1, 2, 3], [4, 5, 6]])
        ff["array2D"] = array_2d
        recovered_2d = ff["array2D"]
        assert np.array_equal(recovered_2d, array_2d)
        
        # Float array
        float_array = np.array([1.1, 2.2, 3.3])
        ff["floatArray"] = float_array
        recovered_float = ff["floatArray"]
        assert np.allclose(recovered_float, float_array)
        
        # Boolean array (should be converted appropriately)
        bool_array = np.array([True, False, True])
        ff["boolArray"] = bool_array
        recovered_bool = ff["boolArray"]
        # Depending on implementation, might be converted to numbers or strings
        assert len(recovered_bool) == len(bool_array)
        
        # Complex structured array
        structured_data = {
            "vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "scalars": [1.0, 2.0, 3.0],
            "matrix": [[1, 2], [3, 4], [5, 6]]
        }
        ff["structuredData"] = structured_data
        
        assert ff["structuredData", "vectors"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert ff["structuredData", "scalars"] == [1.0, 2.0, 3.0]
        assert ff["structuredData", "matrix"] == [[1, 2], [3, 4], [5, 6]]


def test_foamfile_special_value_handling():
    """Test FoamFile handling of special values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "specialValues"
        ff = FoamFile(test_file)
        
        # Test various special values
        special_values = {
            "infinity": float('inf'),
            "negative_infinity": float('-inf'),
            "very_small": 1e-50,
            "very_large": 1e50,
            "negative_zero": -0.0,
            "pi": np.pi,
            "e": np.e
        }
        
        for key, value in special_values.items():
            ff[key] = value
            recovered = ff[key]
            
            if np.isfinite(value):
                assert np.isclose(recovered, value, rtol=1e-14)
            else:
                assert np.isinf(recovered) and np.sign(recovered) == np.sign(value)


def test_field_file_uniform_vs_nonuniform():
    """Test uniform vs nonuniform field handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "uniformTest"
        ff = FoamFieldFile(test_file)
        
        # Test uniform scalar field
        uniform_scalar = 42.0
        ff.internal_field = uniform_scalar
        recovered = ff.internal_field
        assert recovered == uniform_scalar
        
        # Test uniform vector field
        uniform_vector = [1.0, 2.0, 3.0]
        ff.internal_field = uniform_vector
        recovered = ff.internal_field
        assert np.array_equal(recovered, uniform_vector)
        
        # Test non-uniform field (should trigger nonuniform format)
        nonuniform_field = np.random.rand(10, 3)
        ff.internal_field = nonuniform_field
        recovered = ff.internal_field
        assert np.allclose(recovered, nonuniform_field)
        
        # Test boundary between uniform and nonuniform
        # (implementation-dependent threshold)
        small_varying_field = np.array([[1, 0, 0], [1.001, 0, 0], [0.999, 0, 0]])
        ff.internal_field = small_varying_field
        recovered = ff.internal_field
        assert np.allclose(recovered, small_varying_field)


def test_field_file_different_data_types():
    """Test FoamFieldFile with different numpy data types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "dataTypes"
        ff = FoamFieldFile(test_file)
        
        # Test different numpy dtypes
        dtypes_to_test = [
            np.float32,
            np.float64,
            np.int32,
            np.int64
        ]
        
        for dtype in dtypes_to_test:
            test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            ff.internal_field = test_data
            recovered = ff.internal_field
            
            # Should be able to recover the data (possibly with type conversion)
            assert np.allclose(recovered, test_data)


def test_complex_boundary_conditions():
    """Test complex boundary condition scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "complexBC"
        ff = FoamFieldFile(test_file)
        
        # Set up field
        ff.internal_field = np.random.rand(100, 3)
        
        # Complex boundary conditions
        complex_bc = {
            "inlet": {
                "type": "fixedValue",
                "value": "nonuniform List<vector> 10((1 0 0) (1.1 0 0) (0.9 0 0) (1 0.1 0) (1 -0.1 0) (1.2 0 0) (0.8 0 0) (1 0 0.1) (1 0 -0.1) (1.05 0.05 0))"
            },
            "outlet": {
                "type": "pressureInletOutletVelocity",
                "value": "uniform (0 0 0)"
            },
            "walls": {
                "type": "fixedValue", 
                "value": "uniform (0 0 0)"
            },
            "cyclic_1": {
                "type": "cyclic"
            },
            "cyclic_2": {
                "type": "cyclic"
            },
            "symmetryPlane": {
                "type": "symmetryPlane"
            },
            "wedge_1": {
                "type": "wedge"
            },
            "wedge_2": {
                "type": "wedge"  
            }
        }
        
        ff["boundaryField"] = complex_bc
        
        # Test access to complex boundary conditions
        boundary_field = ff.boundary_field
        
        assert boundary_field["inlet"].type == "fixedValue"
        assert "nonuniform" in boundary_field["inlet"].value
        
        assert boundary_field["outlet"].type == "pressureInletOutletVelocity"
        assert boundary_field["cyclic_1"].type == "cyclic"
        assert boundary_field["symmetryPlane"].type == "symmetryPlane"
        
        # Test modification
        boundary_field["walls"].value = "uniform (0.1 0 0)"
        assert boundary_field["walls"].value == "uniform (0.1 0 0)"