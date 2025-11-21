from __future__ import annotations

import gzip
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 10):
    from contextlib import AbstractContextManager
else:
    from typing import ContextManager as AbstractContextManager

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    import os
    from types import TracebackType

from ._parsing import Parsed


class FoamFileIO(AbstractContextManager["FoamFileIO"]):
    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).absolute()

        self.__parsed: Parsed | None = None
        self.__missing: bool | None = None
        self.__defer_io = 0

    @override
    def __enter__(self) -> Self:
        """Read the file from disk if not already read, and defer writing of changes until the context is exited."""
        if self.__defer_io == 0:
            self._get_parsed(missing_ok=True)
        self.__defer_io += 1
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """If this is the outermost context, write any deferred file changes to disk."""
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

    @override
    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"
