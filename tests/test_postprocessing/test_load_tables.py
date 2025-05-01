from __future__ import annotations

import pytest
from foamlib.postprocessing.load_tables import _of_case, of_cases

def test_is_of_case() -> None:
    """Test if a directory is an OpenFOAM case."""
    
    cases = of_cases("tests/test_postprocessing/Cases")
    assert len(cases) == 9
