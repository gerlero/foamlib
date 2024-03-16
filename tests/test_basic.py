import pytest

from foamlib import FoamCase


def test_invalid_case() -> None:
    with pytest.raises(NotADirectoryError):
        FoamCase("invalid_case")
