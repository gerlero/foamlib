import gzip
import sys
import threading
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import os
from types import TracebackType

from ._parsing import ParsedFile


class FoamFileIO(AbstractContextManager["FoamFileIO"]):
    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = Path(path).absolute()

        self.__cached_parsed: ParsedFile | None = None
        self.__file_exists: bool | None = None
        self.__context_depth = 0
        self.__lazy_parse_lock = threading.Lock()

    @override
    def __enter__(self) -> Self:
        """Read the file from disk if not already read, and defer writing of changes until the context is exited."""
        if self.__context_depth == 0:
            self._get_parsed(missing_ok=True, _assume_exclusive=True)
        self.__context_depth += 1
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """If this is the outermost context, write any deferred file changes to disk."""
        try:
            if self.__context_depth == 1:
                assert self.__cached_parsed is not None
                if self.__cached_parsed.modified:
                    contents = self.__cached_parsed.contents

                    if self.path.suffix == ".gz":
                        contents = gzip.compress(contents)

                    self.path.write_bytes(contents)
                    self.__cached_parsed.modified = False
                    self.__file_exists = True
        finally:
            self.__context_depth -= 1
            if (
                self.__context_depth == 0
                and self.__cached_parsed is not None
                and self.__cached_parsed.modified
            ):
                self.__cached_parsed = None

    def _get_parsed(
        self, *, missing_ok: bool = False, _assume_exclusive: bool = False
    ) -> ParsedFile:
        if self.__context_depth == 0:
            with self.__lazy_parse_lock if not _assume_exclusive else nullcontext():
                try:
                    with self.path.open("rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        contents = bytearray(size)
                        f.seek(0)
                        f.readinto(contents)
                except FileNotFoundError:
                    contents = bytearray()
                    self.__file_exists = False
                else:
                    self.__file_exists = True
                    if self.path.suffix == ".gz":
                        contents = bytearray(gzip.decompress(contents))

                assert (
                    self.__cached_parsed is None or not self.__cached_parsed.modified
                ), "cached parsed file had unsaved modifications"

                if (
                    self.__cached_parsed is None
                    or self.__cached_parsed.contents != contents
                ):
                    self.__cached_parsed = ParsedFile(contents)

        assert self.__cached_parsed is not None
        assert self.__file_exists is not None

        if (
            not self.__file_exists
            and not self.__cached_parsed.modified
            and not missing_ok
        ):
            raise FileNotFoundError(self.path)

        return self.__cached_parsed

    @override
    def __repr__(self) -> str:
        return f"{type(self).__qualname__}('{self.path}')"
