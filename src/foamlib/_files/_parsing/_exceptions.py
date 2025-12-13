import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class ParseError(ValueError):
    def __init__(self, contents: bytes, pos: int, *, expected: str) -> None:
        super().__init__()
        self._contents = contents
        self.pos = pos
        self._expected = expected

    @property
    def lineno(self) -> int:
        return self._contents.count(b"\n", 0, self.pos) + 1

    @property
    def colno(self) -> int:
        last_newline = self._contents.rfind(b"\n", 0, self.pos)
        if last_newline == -1:
            return self.pos + 1
        return self.pos - last_newline

    @override
    def __str__(self) -> str:
        snippet_start = max(0, self.pos - 10)
        snippet_end = min(len(self._contents), self.pos + 10)
        snippet = self._contents[snippet_start:snippet_end].decode(
            "ascii", errors="replace"
        )
        pointer = " " * (self.pos - snippet_start) + "^"
        return f"Failure to parse at position {self.pos} (line {self.lineno}, column {self.colno}), Expected {self._expected}:\n{snippet}\n{pointer}"

    @override
    def __repr__(self) -> str:
        return f"ParseError(contents=..., pos={self.pos}, expected={self._expected!r})"
