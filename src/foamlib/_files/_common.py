def _expect_field(keywords: tuple[str, ...], /) -> bool:
    match keywords:
        case ("internalField",):
            return True
        case ("boundaryField", _, k) if k in (
            "value",
            "gradient",
        ) or k.endswith(("Value", "Gradient")):
            return True
    return False


class _FieldKeywords:
    def __eq__(self, keywords: tuple[str, ...], /) -> bool:  # ty: ignore[invalid-method-override]
        return _expect_field(keywords)

    __hash__ = None


FIELD_KEYWORDS = _FieldKeywords()
