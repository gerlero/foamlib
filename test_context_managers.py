#!/usr/bin/env python3
"""
Simple test script to verify context manager functionality
"""
import tempfile
import shutil
from pathlib import Path
import asyncio
from foamlib import FoamCase, AsyncFoamCase

def test_sync_context_manager():
    """Test FoamCase run_context method"""
    # Create a temporary directory to work with
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple case structure
        case_dir = Path(temp_dir) / "test_case"
        case_dir.mkdir()
        
        # Create basic OpenFOAM case structure
        (case_dir / "system").mkdir()
        (case_dir / "constant").mkdir()
        (case_dir / "0").mkdir()
        
        # Create a basic controlDict
        control_dict = case_dir / "system" / "controlDict"
        control_dict.write_text("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     dummy;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          0.1;
writeControl    timeStep;
writeInterval   10;
""")
        
        # Test the context manager
        case = FoamCase(case_dir)
        print(f"Case directory: {case_dir}")
        print(f"Files before run: {list(case_dir.iterdir())}")
        
        # Test that run_context method exists and is callable
        assert hasattr(case, 'run_context'), "FoamCase should have run_context method"
        
        # For this test, we'll just call the method without actually running (to avoid OpenFOAM dependency)
        # We'll test with a simple command that should work
        try:
            with case.run_context(cmd="echo 'test'", log=False) as running_case:
                print("Context manager entered successfully")
                assert running_case is case, "Context manager should return the case instance"
                print("Context manager will clean on exit")
                
            print("Context manager exited successfully")
            print("Test passed!")
            
        except Exception as e:
            print(f"Error: {e}")
            return False
            
    return True

async def test_async_context_manager():
    """Test AsyncFoamCase run_context method"""
    # Create a temporary directory to work with
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple case structure
        case_dir = Path(temp_dir) / "test_case"
        case_dir.mkdir()
        
        # Create basic OpenFOAM case structure
        (case_dir / "system").mkdir()
        (case_dir / "constant").mkdir()
        (case_dir / "0").mkdir()
        
        # Create a basic controlDict
        control_dict = case_dir / "system" / "controlDict"
        control_dict.write_text("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     dummy;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          0.1;
writeControl    timeStep;
writeInterval   10;
""")
        
        # Test the context manager
        case = AsyncFoamCase(case_dir)
        print(f"Async case directory: {case_dir}")
        print(f"Files before run: {list(case_dir.iterdir())}")
        
        # Test that run_context method exists and is callable
        assert hasattr(case, 'run_context'), "AsyncFoamCase should have run_context method"
        
        # For this test, we'll just call the method without actually running (to avoid OpenFOAM dependency)
        # We'll test with a simple command that should work
        try:
            async with case.run_context(cmd="echo 'test'", log=False) as running_case:
                print("Async context manager entered successfully")
                assert running_case is case, "Context manager should return the case instance"
                print("Async context manager will clean on exit")
                
            print("Async context manager exited successfully")
            print("Async test passed!")
            
        except Exception as e:
            print(f"Async error: {e}")
            return False
            
    return True

if __name__ == "__main__":
    print("Testing FoamCase run_context...")
    sync_success = test_sync_context_manager()
    
    print("\nTesting AsyncFoamCase run_context...")
    async_success = asyncio.run(test_async_context_manager())
    
    if sync_success and async_success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)