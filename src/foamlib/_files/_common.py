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
