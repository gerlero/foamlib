"""
Test case for issue #319: FoamFile.update() inserts new line before updated entry

This test ensures that repeated updates to sub-dictionary entries don't accumulate blank lines.
"""

from __future__ import annotations

from pathlib import Path

from foamlib import FoamFile


def test_subdictionary_update_no_blank_line_accumulation(tmp_path: Path) -> None:
    """Test that FoamFile.update() doesn't accumulate blank lines in sub-dictionaries."""

    # Test case from issue #319
    initial_content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.3.0                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    ".";
    object      testDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

subDict
{
    first_line  first_val;

    second_line second_val;

    third_line  third_val;
}

// ************************************************************************* //
"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "file").write_text(initial_content)

    try:
        # Count initial blank lines before second_line
        initial_lines = (tmp_path / "file").read_text().split("\n")

        initial_blank_count = 0
        second_line_idx: int | None = None
        for i, line in enumerate(initial_lines):
            if "second_line" in line and "second_val" in line:
                second_line_idx = i
                # Count blank lines before this line
                for j in range(i - 1, -1, -1):
                    if initial_lines[j].strip() == "":
                        initial_blank_count += 1
                    else:
                        break
                break

        assert second_line_idx is not None, (
            "Could not find second_line in initial content"
        )

        # Perform multiple updates and ensure blank lines don't accumulate
        blank_line_counts: list[int] = []

        for i in range(5):
            testDict = FoamFile(tmp_path / "file")

            # Update using the update() method (as mentioned in the issue)
            updateDict = {"second_line": f"update_val_{i + 1}"}
            subdict = testDict["subDict"]
            assert isinstance(subdict, FoamFile.SubDict)
            subdict.update(updateDict)

            # Count blank lines before second_line after update
            lines = (tmp_path / "file").read_text().split("\n")

            blank_count = 0
            for j, line in enumerate(lines):
                if "second_line" in line and f"update_val_{i + 1}" in line:
                    # Count blank lines before this line
                    for k in range(j - 1, -1, -1):
                        if lines[k].strip() == "":
                            blank_count += 1
                        else:
                            break
                    break

            blank_line_counts.append(blank_count)

        # The key assertion: blank lines should not accumulate
        # Allow for at most one more blank line than initially present (for formatting)
        max_allowed_blanks = initial_blank_count + 1

        for i, count in enumerate(blank_line_counts):
            assert count <= max_allowed_blanks, (
                f"Update {i + 1}: Found {count} blank lines before second_line, "
                f"but expected at most {max_allowed_blanks} (initial: {initial_blank_count}). "
                f"Blank line accumulation detected!"
            )

        # Additionally, ensure that the number of blank lines doesn't keep increasing
        assert all(count <= blank_line_counts[0] + 1 for count in blank_line_counts), (
            f"Blank lines are accumulating: {blank_line_counts}"
        )

    finally:
        (tmp_path / "file").unlink()


def test_subdictionary_direct_assignment_no_blank_line_accumulation(
    tmp_path: Path,
) -> None:
    """Test that direct assignment to sub-dictionary entries doesn't accumulate blank lines."""

    initial_content = """subDict
{
    first_line  first_val;
    second_line second_val;
    third_line  third_val;
}"""

    full_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.3.0                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    ".";
    object      testDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{initial_content}

// ************************************************************************* //
"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "file").write_text(full_content)

    try:
        # Perform multiple direct assignments
        blank_line_counts: list[int] = []

        for i in range(3):
            testDict = FoamFile(tmp_path / "file")

            # Direct assignment (alternative method)
            subdict = testDict["subDict"]
            assert isinstance(subdict, FoamFile.SubDict)
            subdict["second_line"] = f"direct_update_{i + 1}"

            # Count blank lines before second_line
            lines = (tmp_path / "file").read_text().split("\n")

            blank_count = 0
            for j, line in enumerate(lines):
                if "second_line" in line and f"direct_update_{i + 1}" in line:
                    # Count blank lines before this line
                    for k in range(j - 1, -1, -1):
                        if lines[k].strip() == "":
                            blank_count += 1
                        else:
                            break
                    break

            blank_line_counts.append(blank_count)

        # Ensure blank lines don't accumulate
        assert all(count <= 1 for count in blank_line_counts), (
            f"Blank lines are accumulating with direct assignment: {blank_line_counts}"
        )

    finally:
        (tmp_path / "file").unlink()
