import sys
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import (
    Optional,
    Tuple,
    Type,
    Union,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._parsing import Parsed


class FoamFileIO:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).absolute()
        if self.path.is_dir():
            raise IsADirectoryError(self.path)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

        self.__contents: Optional[str] = None
        self.__parsed: Optional[Parsed] = None
        self.__defer_io = 0
        self.__dirty = False

    def __enter__(self) -> Self:
        if self.__defer_io == 0:
            self._read()
        self.__defer_io += 1
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.__defer_io -= 1
        if self.__defer_io == 0 and self.__dirty:
            assert self.__contents is not None
            self._write(self.__contents)
        assert not self.__dirty

    def _read(self) -> Tuple[str, Parsed]:
        if not self.__defer_io:
            contents = self.path.read_text()
            if contents != self.__contents:
                self.__contents = contents
                self.__parsed = None

        assert self.__contents is not None

        if self.__parsed is None:
            parsed = Parsed(self.__contents)
            self.__parsed = parsed

        return self.__contents, deepcopy(self.__parsed)

    def _write(self, contents: str) -> None:
        self.__contents = contents
        self.__parsed = None
        if not self.__defer_io:
            self.path.write_text(contents)
            self.__dirty = False
        else:
            self.__dirty = True
