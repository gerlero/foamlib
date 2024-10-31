import gzip
import sys
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Optional,
    Type,
    Union,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._parsing import Parsed

if TYPE_CHECKING:
    import os


class FoamFileIO:
    def __init__(self, path: Union["os.PathLike[str]", str]) -> None:
        self.path = Path(path).absolute()

        self.__parsed: Optional[Parsed] = None
        self.__missing: Optional[bool] = None
        self.__defer_io = 0

    def __enter__(self) -> Self:
        if self.__defer_io == 0:
            self._get_parsed(missing_ok=True)
        self.__defer_io += 1
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.__defer_io -= 1
        if self.__defer_io == 0:
            assert self.__parsed is not None
            if self.__parsed.modified or self.__missing:
                contents = self.__parsed.contents

                if self.path.suffix == ".gz":
                    contents = gzip.compress(contents)

                self.path.write_bytes(contents)

    def _get_parsed(self, *, missing_ok: bool = False) -> Parsed:
        if not self.__defer_io:
            try:
                contents = self.path.read_bytes()
            except FileNotFoundError:
                self.__missing = True
                contents = b""
            else:
                self.__missing = False
                if self.path.suffix == ".gz":
                    contents = gzip.decompress(contents)

            if self.__parsed is None or self.__parsed.contents != contents:
                self.__parsed = Parsed(contents)

        assert self.__parsed is not None
        assert self.__missing is not None

        if self.__missing and not missing_ok:
            raise FileNotFoundError(self.path)

        return self.__parsed

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"
