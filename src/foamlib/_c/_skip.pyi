"""Type stubs for the _skip C extension module."""

def skip(
    contents: bytes | bytearray,
    pos: int,
    *,
    newline_ok: bool = True,
) -> int:
    """
    Skip whitespace and comments in OpenFOAM file content.
    
    This function skips over whitespace characters (space, tab, newline, etc.) and
    both types of comments:
    - Line comments starting with // (until the end of the line)
    - Block comments enclosed in /* */
    
    Args:
        contents: The byte content to parse (bytes or bytearray).
        pos: The starting position in the content.
        newline_ok: Whether newlines should be considered whitespace (default: True).
                   When False, the function will stop at newline characters.
    
    Returns:
        The new position after skipping whitespace and comments.
    
    Raises:
        FoamFileDecodeError: If an unclosed block comment is encountered.
    """
    ...
