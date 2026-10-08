from foamlib import dumps, loads


def test_issue_730() -> None:
    file = {
        "functions": {
            "difference": {
                "type": "coded",
                "libs": ["utilityFunctionObjects"],
                "name": "writeMagU",
                "codeWrite": '#{\
        const volVectorField& U = mesh().lookupObject<volVectorField>("U");\
        mag(U)().write();\
    #}',
            }
        }
    }

    contents = dumps(file)  # ty: ignore[invalid-argument-type]
    read = loads(contents)
    assert read == file
