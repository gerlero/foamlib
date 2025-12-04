from collections.abc import Iterable, Mapping
from typing import TypeVar, overload

from multicollections import MultiDict

from ._typing import Dict, File, SubDict
from ._util import add_to_mapping

_V = TypeVar("_V")


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[Dict] = ...,
) -> dict[str, _V]: ...


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[File] = ...,
) -> dict[str | None, _V] | MultiDict[str | None, _V]: ...


@overload
def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[SubDict] = ...,
) -> dict[str, _V] | MultiDict[str, _V]: ...


def dict_from_items(
    items: Iterable[tuple[object, _V]],
    /,
    *,
    target: type[Dict] | type[File] | type[SubDict] = Dict,
) -> (
    dict[str, _V]
    | MultiDict[str, _V]
    | dict[str | None, _V]
    | MultiDict[str | None, _V]
):
    none_ok = target is File
    directive_ok = target in (File, SubDict)

    ret: (
        dict[str, _V]
        | MultiDict[str, _V]
        | dict[str | None, _V]
        | MultiDict[str | None, _V]
    ) = {}
    for k, v in items:
        match k:
            case None:
                if not none_ok:
                    msg = "None key is only allowed on top-level File dicts"
                    raise TypeError(msg)
                if None in ret:
                    msg = "Duplicate None key found"
                    raise ValueError(msg)
                ret[None] = v

            case str():
                if k.startswith("#"):
                    if not directive_ok:
                        msg = f"#-directive {k!r} not allowed here"
                        raise ValueError(msg)
                    if isinstance(v, Mapping):
                        msg = f"#-directive {k!r} cannot have a mapping as value; got value {v!r}"
                        raise TypeError(msg)
                    ret = add_to_mapping(ret, k, v)  # ty: ignore[invalid-assignment]
                else:
                    if k in ret:
                        msg = f"Duplicate key found: {k!r}"
                        raise ValueError(msg)
                    ret[k] = v

            case _:
                msg = "Key must be a string"
                if none_ok:
                    msg += " or None"
                msg += f"; got {k!r}"
                raise TypeError(msg)

    return ret
