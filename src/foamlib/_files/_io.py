from __future__ import annotations

import gzip
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._parsing import Parsed

if TYPE_CHECKING:
    import os
    from types import TracebackType


class FoamFileIO:
    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).absolute()

        self.__parsed: Parsed | None = None
        self.__missing: bool | None = None
        self.__defer_io = 0

    def __enter__(self) -> Self:
        if self.__defer_io == 0:
            self._get_parsed(missing_ok=True)
        self.__defer_io += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__defer_io -= 1
        if self.__defer_io == 0:
            assert self.__parsed is not None
            if self.__parsed.modified:
                contents = self.__parsed.contents

                if self.path.suffix == ".gz":
                    contents = gzip.compress(contents)

                self.path.write_bytes(contents)
                self.__parsed.modified = False
                self.__missing = False

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

        if self.__missing and not self.__parsed.modified and not missing_ok:
            raise FileNotFoundError(self.path)

        return self.__parsed

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"
