class ParseError(ValueError):
    """Base class for errors encountered when parsing a FoamFile."""

    def __init__(self, contents: bytes, pos: int) -> None:
        self._contents = contents
        self.pos = pos
        super().__init__()

    @property
    def lineno(self) -> int:
        return self._contents.count(b"\n", 0, self.pos) + 1

    @property
    def colno(self) -> int:
        last_newline = self._contents.rfind(b"\n", 0, self.pos)
        if last_newline == -1:
            return self.pos + 1
        return self.pos - last_newline

    @property
    def _line(self) -> str:
        start = self._contents.rfind(b"\n", 0, self.pos) + 1
        end = self._contents.find(b"\n", self.pos)
        if end == -1:
            end = len(self._contents)
        return self._contents[start:end].decode("ascii", errors="replace")

    @property
    def _column_pointer(self) -> str:
        return " " * (self.colno - 1) + "^"

    def __str__(self) -> str:
        return f"parse failed on line {self.lineno}, column {self.colno}:\n{self._line}\n{self._column_pointer}"

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(line={self.lineno}, column={self.colno})"

    __qualname__ = "foamlib.FoamFile.ParseError"


class ParseSyntaxError(ParseError):
    """Error raised when a FoamFile has unexpected syntax."""

    def __init__(self, contents: bytes, pos: int, *, expected: str) -> None:
        self._expected = expected
        super().__init__(contents, pos)

    def __str__(self) -> str:
        return f"{super().__str__()}\nNote: expected {self._expected}"

    __qualname__ = "foamlib.FoamFile.ParseSyntaxError"


class ParseSemanticError(ParseError):
    """Error raised when a semantic issue is detected in a FoamFile."""

    def __init__(self, contents: bytes, pos: int, *, found: str) -> None:
        self._found = found
        super().__init__(contents, pos)

    def __str__(self) -> str:
        return f"{super().__str__()}\nNote: found {self._found}"

    __qualname__ = "foamlib.FoamFile.ParseSemanticError"
