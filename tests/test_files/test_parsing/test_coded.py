from foamlib import FoamFile


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

    contents = FoamFile.dumps(file)  # ty: ignore[invalid-argument-type]
    read = FoamFile.loads(contents)
    assert read == file
