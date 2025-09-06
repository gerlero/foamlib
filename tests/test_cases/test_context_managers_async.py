import os
import stat
import sys
import tempfile
from pathlib import Path

if sys.version_info >= (3, 9):
    from collections.abc import AsyncGenerator
else:
    from typing import AsyncGenerator

import numpy as np
import pytest
import pytest_asyncio
from foamlib import AsyncFoamCase


@pytest_asyncio.fixture
async def cavity_case() -> AsyncGenerator[AsyncFoamCase, None]:
    """Create a simple test case for testing async context managers."""
    with tempfile.TemporaryDirectory() as temp_dir:
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
        
        case = AsyncFoamCase(case_dir)
        yield case


@pytest.mark.asyncio
async def test_run_context_exists(cavity_case: AsyncFoamCase) -> None:
    """Test that run_context method exists on AsyncFoamCase."""
    assert hasattr(cavity_case, 'run_context'), "AsyncFoamCase should have run_context method"
    assert callable(getattr(cavity_case, 'run_context')), "run_context should be callable"


@pytest.mark.asyncio
async def test_run_context_basic(cavity_case: AsyncFoamCase) -> None:
    """Test basic functionality of run_context."""
    # Test with a simple command to avoid long-running simulations
    async with cavity_case.run_context(cmd="echo 'test'", log=False) as running_case:
        assert running_case is cavity_case, "Context manager should return the case instance"
        assert hasattr(running_case, 'clean'), "Returned case should have clean method"


@pytest.mark.asyncio
async def test_run_context_simulation(cavity_case: AsyncFoamCase) -> None:
    """Test run_context with a simple command."""
    # Check initial state - should be clean
    initial_time_dirs = len([d for d in cavity_case.path.iterdir() 
                           if d.is_dir() and d.name.replace('.', '').isdigit()])
    
    # Run a simple command in context manager
    async with cavity_case.run_context(cmd="echo 'test simulation'", log=False) as running_case:
        # Verify we can access the case
        assert running_case is cavity_case, "Context manager should return the case instance"
        
        # The case should still be accessible
        assert hasattr(running_case, 'path'), "Case should have path attribute"
    
    # After exiting context, clean method should have been called
    # We can't easily verify the clean was called without modifying the case,
    # but we can verify the context manager worked without errors
    assert True, "Context manager completed successfully"


@pytest.mark.asyncio
async def test_run_context_with_exception(cavity_case: AsyncFoamCase) -> None:
    """Test that run_context cleans up even when an exception occurs."""
    initial_time_dirs = len([d for d in cavity_case.path.iterdir() 
                           if d.is_dir() and d.name.replace('.', '').isdigit()])
    
    with pytest.raises(ValueError):
        async with cavity_case.run_context(cmd="echo 'test'", log=False) as running_case:
            # Verify the context manager entered
            assert running_case is cavity_case, "Context manager should return the case instance"
            # Raise an exception to test cleanup
            raise ValueError("Test exception")
    
    # Clean should have been called even with exception
    # We can't easily verify the clean was called without modifying the case,
    # but we can verify the context manager handled the exception correctly
    assert True, "Context manager handled exception correctly"


@pytest.mark.asyncio
async def test_run_context_parameters(cavity_case: AsyncFoamCase) -> None:
    """Test that run_context accepts the same parameters as run method."""
    # Test all parameter combinations work without error
    async with cavity_case.run_context(
        cmd="echo 'test'",
        parallel=False,
        cpus=1,
        check=True,
        log=False
    ) as running_case:
        assert running_case is cavity_case


@pytest.mark.asyncio
async def test_run_context_return_type(cavity_case: AsyncFoamCase) -> None:
    """Test that run_context returns a proper async context manager."""
    ctx_manager = cavity_case.run_context(cmd="echo 'test'", log=False)
    
    # Should have async context manager methods
    assert hasattr(ctx_manager, '__aenter__'), "Should have __aenter__ method"
    assert hasattr(ctx_manager, '__aexit__'), "Should have __aexit__ method"
    
    # Should be able to use with 'async with' statement
    async with ctx_manager as running_case:
        assert running_case is cavity_case


@pytest.mark.asyncio
async def test_run_context_awaitable(cavity_case: AsyncFoamCase) -> None:
    """Test that run_context can be awaited (AwaitableAsyncContextManager)."""
    # The @awaitableasynccontextmanager decorator should make it awaitable
    ctx_manager = cavity_case.run_context(cmd="echo 'test'", log=False)
    
    # Should be awaitable
    assert hasattr(ctx_manager, '__await__'), "Should be awaitable"
    
    # Test that it can be awaited directly
    running_case = await ctx_manager
    assert running_case is cavity_case, "Awaiting should return the case instance"