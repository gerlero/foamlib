from foamlib import FoamFile


def test_issue_730() -> None:
    file = {}
    file["functions"] = {}
    file["functions"]["difference"] = {
        "type": "coded",
        "libs": ["utilityFunctionObjects"],
        "name": "writeMagU",
        "codeWrite": '#{\
        const volVectorField& U = mesh().lookupObject<volVectorField>("U");\
        mag(U)().write();\
    #}',
    }

    contents = FoamFile.dumps(file)
    read = FoamFile.loads(contents)
    assert read == file
