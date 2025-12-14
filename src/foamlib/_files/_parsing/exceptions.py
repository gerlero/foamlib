class FoamFileDecodeError(ValueError):
    """Error raised when a FoamFile cannot be parsed."""

    def __init__(self, contents: bytes | bytearray, pos: int, *, expected: str) -> None:
        self._contents = contents
        self.pos = pos
        self._expected = expected
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
        return f"parsing failed on line {self.lineno}, column {self.colno}:\n{self._line}\n{self._column_pointer}\nExpected: {self._expected}"

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(line={self.lineno}, column={self.colno})"

    __name__ = "FoamFileDecodeError"
    __qualname__ = "foamlib.FoamFileDecodeError"
