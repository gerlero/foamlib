import sys
from typing import (
    Any,
    Tuple,
    Union,
)

if sys.version_info >= (3, 9):
    from collections.abc import Iterator, Mapping, MutableMapping
else:
    from typing import Iterator, Mapping, MutableMapping

from ._base import FoamDict
from ._io import FoamFileIO
from ._serialization import serialize_keyword_entry


class FoamFile(
    FoamDict,
    MutableMapping[
        Union[str, Tuple[str, ...]], Union["FoamFile.Data", "FoamFile.SubDict"]
    ],
    FoamFileIO,
):
    """
    An OpenFOAM dictionary file.

    Use as a mutable mapping (i.e., like a dict) to access and modify entries.

    Use as a context manager to make multiple changes to the file while saving all changes only once at the end.
    """

    class SubDict(
        FoamDict,
        MutableMapping[str, Union["FoamFile.Data", "FoamFile.SubDict"]],
    ):
        """An OpenFOAM dictionary within a file as a mutable mapping."""

        def __init__(self, _file: "FoamFile", _keywords: Tuple[str, ...]) -> None:
            self._file = _file
            self._keywords = _keywords

        def __getitem__(
            self, keyword: str
        ) -> Union["FoamFile.Data", "FoamFile.SubDict"]:
            return self._file[(*self._keywords, keyword)]

        def _setitem(
            self,
            keyword: str,
            data: Any,
            *,
            assume_field: bool = False,
            assume_dimensions: bool = False,
        ) -> None:
            self._file._setitem(
                (*self._keywords, keyword),
                data,
                assume_field=assume_field,
                assume_dimensions=assume_dimensions,
            )

        def __setitem__(self, keyword: str, value: "FoamFile._SetData") -> None:
            self._setitem(keyword, value)

        def __delitem__(self, keyword: str) -> None:
            del self._file[(*self._keywords, keyword)]

        def __iter__(self) -> Iterator[str]:
            return self._file._iter(self._keywords)

        def __contains__(self, keyword: object) -> bool:
            return (*self._keywords, keyword) in self._file

        def __len__(self) -> int:
            return len(list(iter(self)))

        def update(self, *args: Any, **kwargs: Any) -> None:
            with self._file:
                super().update(*args, **kwargs)

        def clear(self) -> None:
            with self._file:
                super().clear()

        def __repr__(self) -> str:
            return f"{type(self).__qualname__}({self._file}, {self._keywords})"

        def as_dict(self) -> FoamDict._Dict:
            """Return a nested dict representation of the dictionary."""
            ret = self._file.as_dict()

            for k in self._keywords:
                assert isinstance(ret, dict)
                v = ret[k]
                assert isinstance(v, dict)
                ret = v

            return ret

    def __getitem__(
        self, keywords: Union[str, Tuple[str, ...]]
    ) -> Union["FoamFile.Data", "FoamFile.SubDict"]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        _, parsed = self._read()

        value = parsed[keywords]

        if value is ...:
            return FoamFile.SubDict(self, keywords)
        else:
            return value  # type: ignore [return-value]

    def _setitem(
        self,
        keywords: Union[str, Tuple[str, ...]],
        data: "FoamFile._SetData",
        *,
        assume_field: bool = False,
        assume_dimensions: bool = False,
    ) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents, parsed = self._read()

        if isinstance(data, Mapping):
            with self:
                if isinstance(data, FoamDict):
                    data = data.as_dict()

                start, end = parsed.entry_location(keywords, missing_ok=True)

                self._write(
                    f"{contents[:start]}\n{serialize_keyword_entry(keywords[-1], {})}\n{contents[end:]}"
                )

                for k, v in data.items():
                    self[(*keywords, k)] = v
        else:
            start, end = parsed.entry_location(keywords, missing_ok=True)

            self._write(
                f"{contents[:start]}\n{serialize_keyword_entry(keywords[-1], data, assume_field=assume_field, assume_dimensions=assume_dimensions)}\n{contents[end:]}"
            )

    def __setitem__(
        self,
        keywords: Union[str, Tuple[str, ...]],
        data: "FoamFile._SetData",
    ) -> None:
        self._setitem(keywords, data)

    def __delitem__(self, keywords: Union[str, Tuple[str, ...]]) -> None:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        contents, parsed = self._read()

        start, end = parsed.entry_location(keywords)

        self._write(contents[:start] + contents[end:])

    def _iter(self, keywords: Union[str, Tuple[str, ...]] = ()) -> Iterator[str]:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)

        _, parsed = self._read()

        yield from (k[-1] for k in parsed if k[:-1] == keywords)

    def __iter__(self) -> Iterator[str]:
        return self._iter()

    def __contains__(self, keywords: object) -> bool:
        if not isinstance(keywords, tuple):
            keywords = (keywords,)
        _, parsed = self._read()
        return keywords in parsed

    def __len__(self) -> int:
        return len(list(iter(self)))

    def update(self, *args: Any, **kwargs: Any) -> None:
        with self:
            super().update(*args, **kwargs)

    def clear(self) -> None:
        with self:
            super().clear()

    def __fspath__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path})"

    def as_dict(self) -> FoamDict._Dict:
        """Return a nested dict representation of the file."""
        _, parsed = self._read()
        return parsed.as_dict()
