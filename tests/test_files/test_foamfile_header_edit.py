"""Test for properly formatted dictionary edits, especially FoamFile header edits."""

from pathlib import Path

from foamlib import FoamFile


def test_foamfile_header_edit_preserves_formatting(tmp_path: Path) -> None:
    """Test that editing FoamFile header entries preserves indentation and newlines."""
    test_file = tmp_path / "testDict"

    # Create a simple FoamFile with a header
    f = FoamFile(test_file)
    f["key1"] = "value1"

    # Check initial formatting
    content = test_file.read_text()
    assert "    version 2.0;" in content
    assert "    format ascii;" in content
    assert "    class dictionary;" in content

    # Edit version
    f["FoamFile", "version"] = 2.1
    content = test_file.read_text()
    assert "    version 2.1;" in content
    assert "    format ascii;" in content
    # Verify no formatting issues (entries should not be on same line)
    lines = content.split("\n")
    version_line = next(line for line in lines if "version" in line)
    assert "format" not in version_line, (
        "version and format should not be on the same line"
    )

    # Edit format
    f["FoamFile", "format"] = "binary"
    content = test_file.read_text()
    assert "    format binary;" in content
    lines = content.split("\n")
    format_line = next(line for line in lines if "format" in line)
    assert "class" not in format_line, "format and class should not be on the same line"

    # Edit class
    f["FoamFile", "class"] = "volScalarField"
    content = test_file.read_text()
    assert "    class volScalarField;" in content

    # Edit object
    f["FoamFile", "object"] = "myDict"
    content = test_file.read_text()
    assert "    object myDict;" in content

    # Verify the overall structure is still correct
    assert content.startswith("FoamFile\n{")
    assert "key1 value1;" in content


def test_subdictionary_edit_preserves_formatting(tmp_path: Path) -> None:
    """Test that editing subdictionary entries preserves indentation and newlines."""
    test_file = tmp_path / "testDict"

    # Create a file with a subdictionary
    f = FoamFile(test_file)
    f["mysubdict"] = {"subkey1": "value1", "subkey2": "value2"}

    # Check initial formatting
    content = test_file.read_text()
    assert "    subkey1 value1;" in content
    assert "    subkey2 value2;" in content

    # Edit subkey1
    f["mysubdict", "subkey1"] = "newvalue1"
    content = test_file.read_text()
    assert "    subkey1 newvalue1;" in content

    # Verify no formatting issues
    lines = content.split("\n")
    subkey1_line = next(line for line in lines if "subkey1" in line)
    assert "subkey2" not in subkey1_line, (
        "subkey1 and subkey2 should not be on the same line"
    )
    assert subkey1_line.startswith("    "), "subkey1 should have proper indentation"

    # Edit subkey2
    f["mysubdict", "subkey2"] = "newvalue2"
    content = test_file.read_text()
    assert "    subkey2 newvalue2;" in content

    # Verify the overall structure
    assert "mysubdict\n{" in content
    assert content.count("    subkey1 newvalue1;") == 1
    assert content.count("    subkey2 newvalue2;") == 1


def test_multiple_edits_preserve_formatting(tmp_path: Path) -> None:
    """Test that multiple consecutive edits maintain proper formatting."""
    test_file = tmp_path / "testDict"

    # Create a file
    f = FoamFile(test_file)
    f["key1"] = "value1"

    # Edit the same FoamFile header entry multiple times to ensure
    # formatting is preserved across multiple updates
    NUM_EDITS = 5
    VERSION_INCREMENT = 0.1

    for i in range(NUM_EDITS):
        f["FoamFile", "version"] = 2.0 + i * VERSION_INCREMENT
        content = test_file.read_text()
        # Verify indentation is preserved
        assert f"    version {2.0 + i * VERSION_INCREMENT};" in content
        # Verify entries are on separate lines
        lines = content.split("\n")
        version_line = next(line for line in lines if "version" in line)
        assert "format" not in version_line
        assert version_line.startswith("    ")


def test_nested_subdictionary_edit_preserves_formatting(tmp_path: Path) -> None:
    """Test that editing nested subdictionary entries preserves indentation."""
    test_file = tmp_path / "testDict"

    # Create a file with nested subdictionaries
    f = FoamFile(test_file)
    f["level1"] = {"level2": {"key": "value"}}

    # Check initial formatting - the initial content has some blank lines which is expected
    content = test_file.read_text()
    # The key should have proper indentation (at least 4 spaces for being inside level2)
    assert "key value;" in content

    # Edit the nested entry
    f["level1", "level2", "key"] = "newvalue"
    content = test_file.read_text()
    assert "key newvalue;" in content

    # Verify the key maintains some indentation
    lines = content.split("\n")
    key_line = next(line for line in lines if "key" in line and "newvalue" in line)
    # The key should start with at least some whitespace (4 spaces minimum)
    assert key_line.startswith("    "), (
        "nested key should have proper indentation (at least 4 spaces)"
    )

    # Verify the closing braces are present
    assert "}" in content
    assert content.count("}") >= 2, "should have closing braces for both nested dicts"
