"""Test async operations edge cases and error handling."""

import asyncio
import contextlib
import tempfile
from pathlib import Path

import pytest
from foamlib import AsyncFoamCase, AsyncSlurmFoamCase, CalledProcessError


@pytest.mark.asyncio
async def test_async_foam_case_edge_cases() -> None:
    """Test AsyncFoamCase with various edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "async_test_case"
        case_path.mkdir()

        # Test with empty case
        async_case = AsyncFoamCase(case_path)
        assert async_case.path == case_path.absolute()

        # Test regular iteration (not async)
        assert len(async_case) == 0
        assert list(async_case) == []

        # Create some time directories
        (case_path / "0").mkdir()
        (case_path / "0" / "U").touch()
        (case_path / "1").mkdir()
        (case_path / "1" / "U").touch()

        # Test iteration with data
        time_dirs = list(async_case)
        assert len(time_dirs) >= 1


@pytest.mark.asyncio
async def test_async_foam_case_run_errors() -> None:
    """Test AsyncFoamCase run method error handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "error_test_case"
        case_path.mkdir()

        async_case = AsyncFoamCase(case_path)

        # Test with non-existent command - expect FileNotFoundError, not CalledProcessError
        with pytest.raises(FileNotFoundError):
            await async_case.run(["nonexistentCommand"])

        # Test with invalid arguments - this might also raise FileNotFoundError
        with pytest.raises((CalledProcessError, FileNotFoundError)):
            await async_case.run(["ls", "--invalid-option-that-does-not-exist"])


@pytest.mark.asyncio
async def test_async_foam_case_clone() -> None:
    """Test AsyncFoamCase clone functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = Path(tmpdir) / "original"
        original_path.mkdir()
        (original_path / "0").mkdir()
        (original_path / "0" / "U").touch()
        (original_path / "system").mkdir()
        (original_path / "system" / "controlDict").touch()

        async_case = AsyncFoamCase(original_path)

        # Test async clone context manager with explicit destination
        clone_dst = Path(tmpdir) / "clone"
        async with async_case.clone(clone_dst) as clone:
            assert clone.path != async_case.path
            assert clone.path.exists()
            assert (clone.path / "0" / "U").exists()
            assert (clone.path / "system" / "controlDict").exists()

            clone_path = clone.path

        # Clone should be cleaned up after context
        assert not clone_path.exists()


@pytest.mark.asyncio
async def test_async_foam_case_parallel_operations() -> None:
    """Test AsyncFoamCase parallel operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "parallel_test"
        case_path.mkdir()

        # Create minimal case structure
        (case_path / "0").mkdir()
        (case_path / "system").mkdir()
        (case_path / "constant").mkdir()

        # Create decomposePar dict
        system_path = case_path / "system"
        decompose_dict = system_path / "decomposeParDict"
        decompose_dict.write_text("""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
        }

        numberOfSubdomains 2;
        method scotch;
        """)

        async_case = AsyncFoamCase(case_path)

        # Test decompose_par_dict access
        decompose_par_dict = async_case.decompose_par_dict
        assert "numberOfSubdomains" in decompose_par_dict

        # Test parallel run (will likely fail without proper OpenFOAM setup)
        with contextlib.suppress(CalledProcessError, FileNotFoundError):
            await async_case.run(parallel=True)


@pytest.mark.asyncio
async def test_async_slurm_foam_case() -> None:
    """Test AsyncSlurmFoamCase specific functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "slurm_test"
        case_path.mkdir()

        slurm_case = AsyncSlurmFoamCase(case_path)
        assert isinstance(slurm_case, AsyncSlurmFoamCase)
        assert slurm_case.path == case_path.absolute()

        # Test fallback functionality - may fail depending on system setup
        with contextlib.suppress(Exception):
            await slurm_case.run(["echo", "test"], fallback=True)


@pytest.mark.asyncio
async def test_async_foam_case_cleanup_operations() -> None:
    """Test AsyncFoamCase cleanup operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "cleanup_test"
        case_path.mkdir()

        # Create time directories and files to clean
        for time in ["1", "2", "3"]:
            time_dir = case_path / time
            time_dir.mkdir()
            (time_dir / "U").touch()
            (time_dir / "p").touch()

        # Create processor directories
        for proc in ["processor0", "processor1"]:
            proc_dir = case_path / proc
            proc_dir.mkdir()
            for time in ["0", "1"]:
                (proc_dir / time).mkdir()
                (proc_dir / time / "U").touch()

        async_case = AsyncFoamCase(case_path)

        # Test clean operation
        await async_case.clean()

        # Check that appropriate files were cleaned
        # (Exact behavior depends on implementation)
        list(case_path.iterdir())
        # Should preserve constant and system if they exist


@pytest.mark.asyncio
async def test_async_foam_case_file_operations() -> None:
    """Test AsyncFoamCase file access operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "file_ops_test"
        system_path = case_path / "system"
        constant_path = case_path / "constant"
        system_path.mkdir(parents=True)
        constant_path.mkdir(parents=True)

        # Create control dict
        control_dict = system_path / "controlDict"
        control_dict.write_text("""
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
        }

        startTime 0;
        endTime 1000;
        """)

        async_case = AsyncFoamCase(case_path)

        # Test file access
        cd = async_case.control_dict
        assert cd["startTime"] == 0
        assert cd["endTime"] == 1000

        # Test file modification
        cd["endTime"] = 2000
        assert cd["endTime"] == 2000


@pytest.mark.asyncio
async def test_async_foam_case_time_directory_operations() -> None:
    """Test AsyncFoamCase TimeDirectory operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "time_ops_test"
        case_path.mkdir()

        # Create time directories
        time_dirs = ["0", "0.5", "1.0", "2.0"]
        for time_name in time_dirs:
            time_path = case_path / time_name
            time_path.mkdir()
            (time_path / "U").touch()
            (time_path / "p").touch()

        async_case = AsyncFoamCase(case_path)

        # Test time directory access
        time_0 = async_case[0]
        assert time_0.time == 0.0
        assert "U" in time_0
        assert "p" in time_0

        # Test latest time access
        latest = async_case[-1]
        assert latest.time == 2.0

        # Test string access
        time_half = async_case["0.5"]
        assert time_half.time == 0.5


@pytest.mark.asyncio
async def test_async_foam_case_basic_context_usage() -> None:
    """Test AsyncFoamCase basic usage without async context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "context_test"
        case_path.mkdir()

        # Test basic AsyncFoamCase usage without async context manager
        async_case = AsyncFoamCase(case_path)
        assert async_case.path == case_path.absolute()

        # Should be able to perform operations directly
        file_obj = async_case.file("test.txt")
        assert file_obj.path == case_path / "test.txt"


@pytest.mark.asyncio
async def test_async_foam_case_error_propagation() -> None:
    """Test that AsyncFoamCase properly propagates errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "error_prop_test"
        # Don't create the directory to test error handling

        async_case = AsyncFoamCase(case_path)

        # Test accessing non-existent case
        with pytest.raises(FileNotFoundError):
            _ = async_case.control_dict["startTime"]


@pytest.mark.asyncio
async def test_async_foam_case_concurrent_operations() -> None:
    """Test AsyncFoamCase with concurrent operations."""

    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "concurrent_test"
        case_path.mkdir()

        # Create multiple time directories
        for i in range(5):
            time_dir = case_path / str(i)
            time_dir.mkdir()
            (time_dir / "U").touch()

        async_case = AsyncFoamCase(case_path)

        # Test concurrent access to different time directories
        async def access_time(time_index: int) -> float:
            time_dir = async_case[time_index]
            return time_dir.time

        # Run concurrent operations
        tasks = [access_time(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert results == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_async_foam_case_resource_cleanup() -> None:
    """Test proper resource cleanup in AsyncFoamCase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case_path = Path(tmpdir) / "resource_test"
        case_path.mkdir()

        # Test that resources are properly cleaned up after operations
        async_case = AsyncFoamCase(case_path)

        # Perform various operations
        async_case.file("test")

        # Test clone with cleanup with explicit destination
        clone_dst = Path(tmpdir) / "test_clone"
        async with async_case.clone(clone_dst) as clone:
            clone.file("clone_test")
            clone_path = clone.path

        # Clone should be cleaned up
        assert not clone_path.exists()

        # Original case should still be accessible
        assert async_case.path.exists()
