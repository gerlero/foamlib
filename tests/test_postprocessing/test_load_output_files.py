from foamlib.postprocessing.tableReader import TableReader

def test_read_forces() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/forces/0/force.dat")
    assert df.shape == (109, 10)
    assert list(df.columns) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_read_freesurface_p() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/p_freeSurface.raw")
    assert df.shape == (278, 4)
    assert list(df.columns) == [0, 1, 2, 3]

def test_read_freesurface_u() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/U_freeSurface.raw")
    assert df.shape == (278, 6)
    assert list(df.columns) == [0, 1, 2, 3, 4, 5]

def test_probes_p() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/p")
    assert df.shape == (109, 2)
    assert list(df.columns) == [0, 1]

def test_probes_U() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/U")
    print(df)
    assert df.shape == (109, 4)
    assert list(df.columns) == [0, 1, 2, 3]

def test_probes_T() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/T")
    assert df.shape == (109, 2)
    assert list(df.columns) == [0, 1]

def test_read_sample_xy() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/sample1/0.1/centreLine_T.xy")
    assert df.shape == (100, 2)
    assert list(df.columns) == [0, 1]