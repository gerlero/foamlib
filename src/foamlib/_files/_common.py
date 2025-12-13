from collections.abc import Iterable, Mapping
from typing import TypeVar, overload

from multicollections import MultiDict

from ..typing import Dict, FileDict, SubDict
from ._util import add_to_mapping

_V = TypeVar("_V")


def _expect_field(keywords: object) -> bool:
    match keywords:
        case ("internalField",):
            return True
        case ("boundaryField", str(), str() as kw) if kw in (
            "value",
            "gradient",
        ) or kw.endswith(("Value", "Gradient")):
            return True
    return False


class _FieldKeywords:
    def __eq__(self, keywords: object) -> bool:
        return _expect_field(keywords)

    __hash__ = None


FIELD_KEYWORDS = _FieldKeywords()


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[Dict] = ...,
    check_keys: bool = ...,
) -> dict[str, _V]: ...


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[SubDict] = ...,
    check_keys: bool = ...,
) -> dict[str, _V] | MultiDict[str, _V]: ...


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[FileDict] = ...,
    check_keys: bool = ...,
) -> dict[str | None, _V] | MultiDict[str | None, _V]: ...


def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[Dict] | type[SubDict] | type[FileDict],
    check_keys: bool = False,
) -> (
    dict[str, _V]
    | MultiDict[str, _V]
    | dict[str | None, _V]
    | MultiDict[str | None, _V]
):
    none_ok = target is FileDict
    directive_ok = target in (FileDict, SubDict)

    ret: (
        dict[str, _V]
        | MultiDict[str, _V]
        | dict[str | None, _V]
        | MultiDict[str | None, _V]
    ) = {}
    for k, v in items:
        match k:
            case None:
                if check_keys:
                    if not none_ok:
                        msg = "None key is only allowed in top-level File dicts"
                        raise TypeError(msg)
                    if None in ret:
                        msg = "duplicate None key found"
                        raise ValueError(msg)
                ret[None] = v

            case str():
                if check_keys:
                    from ._parsing import parse  # noqa: PLC0415

                    if k != parse(k, target=str):
                        msg = f"invalid key string: {k!r}"
                        raise ValueError(msg)
                if k.startswith("#"):
                    if check_keys:
                        if not directive_ok:
                            msg = f"#-directive {k!r} not allowed here"
                            raise ValueError(msg)
                        if isinstance(v, Mapping):
                            msg = f"#-directive {k!r} cannot have a mapping as value; got value {v!r}"
                            raise TypeError(msg)
                    ret = add_to_mapping(ret, k, v)  # ty: ignore[invalid-assignment]
                else:
                    if k in ret:
                        msg = f"duplicate key found: {k!r}"
                        raise ValueError(msg)
                    ret[k] = v

            case _:
                msg = "key must be a string"
                if none_ok:
                    msg += " or None"
                msg += f"; got {k!r}"
                raise TypeError(msg)

    return ret
