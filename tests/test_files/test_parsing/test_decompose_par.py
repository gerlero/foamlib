# Based on https://foss.heptapod.net/fluiddyn/fluidsimfoam/-/blob/branch/default/tests/test_decompose_par.py

from pathlib import Path
from textwrap import dedent

import numpy as np
from foamlib import FoamFile


def test_simple(tmp_path: Path) -> None:
    contents = dedent(
        """
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            location    "system";
            object      decomposeParDict;
        }

        numberOfSubdomains  8;

        method            simple;

        coeffs
        {
            n
            (
                4
                2
                1
            );
            order    xyz;
            delta    0.001;
        }
    """
    ).strip()

    path = tmp_path / "decomposeParDict"
    path.write_text(contents)

    decompose_par_dict = FoamFile(path)

    assert decompose_par_dict["numberOfSubdomains"] == 8
    assert decompose_par_dict["method"] == "simple"
    assert np.array_equal(decompose_par_dict["coeffs", "n"], [4, 2, 1])  # type: ignore[arg-type]
    assert decompose_par_dict["coeffs", "order"] == "xyz"
    assert decompose_par_dict["coeffs", "delta"] == 0.001


def test_scotch(tmp_path: Path) -> None:
    contents = dedent(
        """
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       dictionary;
            location    "system";
            object      decomposeParDict;
        }

        numberOfSubdomains  12;

        method            scotch;
    """
    ).strip()

    path = tmp_path / "decomposeParDict"
    path.write_text(contents)

    decompose_par_dict = FoamFile(path)
    assert decompose_par_dict["numberOfSubdomains"] == 12
    assert decompose_par_dict["method"] == "scotch"
