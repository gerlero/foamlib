"""Test for preserving trailing newline when modifying subdictionary entries."""

from pathlib import Path

from foamlib import FoamFile


def test_modify_subdictionary_entry_preserves_trailing_newline(tmp_path: Path) -> None:
    """Test that modifying a subdictionary entry preserves the trailing newline."""
    test_file = tmp_path / "testDict"

    # Create a file with subdictionary entries
    content = """LeungLindstedtJonesCoeffs
{
    Cbeta       Cbeta   [0 0 0 0 0 0 0]  3;
    bBeta       bBeta   [0 0.5 -1 0 0 0 0]      10.0e4;
    nBeta       nBeta   [0 0 0 0 0 0 0]  0.5;
}"""
    test_file.write_text(content)

    # Modify bBeta entry
    with FoamFile(test_file) as f:
        f["LeungLindstedtJonesCoeffs"]["bBeta"] = f.Dimensioned(
            1.2e4, [0, 0.5, -1, 0, 0, 0, 0], "bBeta"
        )

    # Check that entries remain on separate lines
    result = test_file.read_text()
    lines = result.split("\n")

    # Find lines with Cbeta and bBeta
    cbeta_line_idx = next(i for i, line in enumerate(lines) if "Cbeta" in line)
    bbeta_line_idx = next(i for i, line in enumerate(lines) if "bBeta" in line)

    # They should be on different lines
    assert cbeta_line_idx != bbeta_line_idx, (
        "Cbeta and bBeta should be on different lines"
    )
    assert bbeta_line_idx == cbeta_line_idx + 1, (
        "bBeta should be on the line immediately after Cbeta"
    )

    # Verify the Cbeta line doesn't contain bBeta
    assert "bBeta" not in lines[cbeta_line_idx], (
        "Cbeta line should not contain bBeta"
    )


def test_modify_subdictionary_entry_with_comments(tmp_path: Path) -> None:
    """Test that modifying entries preserves formatting with comments."""
    test_file = tmp_path / "sootCoeffs"

    # Exact content from the issue
    content = """sootModel LeungLindstedtJones;
LeungLindstedtJonesCoeffs
{
    // ...
    aAlpha      aAlpha  [0 0 -1 0 0 0 0]    0.1e5; // Pre-exponential factor
    TaAlpha     TaAlpha [0 0 0 1 0 0 0] 21100; // Activation temperature


    Cbeta       Cbeta   [0 0 0 0 0 0 0]  3; // Agglomeration rate constant
    bBeta       bBeta   [0 0.5 -1 0 0 0 0]      10.0e4; // Pre-exponential factor
    nBeta       nBeta   [0 0 0 0 0 0 0]  0.5; // ??
    TaBeta      TaBeta  [0 0 0 1 0 0 0] 12100; // Activation temperature
    // ...
}"""
    test_file.write_text(content)

    # Modify bBeta entry
    with FoamFile(test_file) as f:
        f["LeungLindstedtJonesCoeffs"]["bBeta"] = f.Dimensioned(
            1.2e4, [0, 0.5, -1, 0, 0, 0, 0], "bBeta"
        )

    # Check result
    result = test_file.read_text()
    lines = result.split("\n")

    # Find the Cbeta line (specifically the one with the keyword, not just containing beta)
    cbeta_line_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Cbeta"))

    # Verify Cbeta and bBeta are on separate lines
    cbeta_line = lines[cbeta_line_idx]
    assert "bBeta" not in cbeta_line, (
        "The line with Cbeta should not also contain bBeta"
    )

    # Verify the modified bBeta entry is on the next line and preserves comment
    bbeta_line = lines[cbeta_line_idx + 1]
    assert "bBeta" in bbeta_line, "bBeta should be on the next line"
    assert "Pre-exponential factor" in bbeta_line, "Comment should be preserved"


def test_multiple_subdictionary_entry_modifications(tmp_path: Path) -> None:
    """Test that multiple consecutive modifications preserve newlines."""
    test_file = tmp_path / "testDict"

    content = """myDict
{
    key1    value1;
    key2    value2;
    key3    value3;
    key4    value4;
}"""
    test_file.write_text(content)

    # Modify multiple entries
    with FoamFile(test_file) as f:
        f["myDict"]["key2"] = "newvalue2"
        f["myDict"]["key3"] = "newvalue3"

    # Check that all entries remain on separate lines
    result = test_file.read_text()
    # Filter to get only entry lines (skip dict name, braces, empty lines)
    lines = result.split("\n")
    entry_lines = [
        line for line in lines
        if line.strip()
        and "{" not in line
        and "}" not in line
        and "myDict" not in line
    ]

    # Should have 4 lines (one for each key)
    assert len(entry_lines) == 4, f"Should have 4 entry lines, got {len(entry_lines)}"

    # Each key should be on its own line (no two keys on the same line)
    for key in ["key1", "key2", "key3", "key4"]:
        # Find lines containing this key
        key_lines = [line for line in entry_lines if key in line]
        assert len(key_lines) == 1, f"{key} should appear in exactly one line"
        # Verify this line doesn't contain other keys
        other_keys = [k for k in ["key1", "key2", "key3", "key4"] if k != key]
        assert not any(
            other_key in key_lines[0] for other_key in other_keys
        ), f"{key} line should not contain other keys"


def test_nested_subdictionary_entry_preserves_newline(tmp_path: Path) -> None:
    """Test that modifying nested subdictionary entries preserves newlines."""
    test_file = tmp_path / "testDict"

    content = """level1
{
    level2
    {
        key1    value1;
        key2    value2;
        key3    value3;
    }
}"""
    test_file.write_text(content)

    # Modify a nested entry
    with FoamFile(test_file) as f:
        f["level1", "level2", "key2"] = "newvalue2"

    # Check that entries remain on separate lines
    result = test_file.read_text()
    lines = result.split("\n")

    # Find lines with keys
    key1_line_idx = next(i for i, line in enumerate(lines) if "key1" in line)
    key2_line_idx = next(i for i, line in enumerate(lines) if "key2" in line)
    key3_line_idx = next(i for i, line in enumerate(lines) if "key3" in line)

    # They should all be on different lines
    assert len({key1_line_idx, key2_line_idx, key3_line_idx}) == 3, (
        "All keys should be on different lines"
    )
