from foamlib.postprocessing.tableReader import TableReader

def test_read_forces() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/forces/0/force.dat")
    assert df.shape == (109, 10)
    assert list(df.columns) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = reader.read("tests/test_postprocessing/postProcessing/forces/0/force.dat", column_names=["time", "Fx", "Fy", "Fz", "p_Fx", "p_Fy", "p_Fz", "visk_Fx", "visk_Fy", "visk_Fz"])
    assert df.shape == (109, 10)
    assert list(df.columns) == ["time", "Fx", "Fy", "Fz", "p_Fx", "p_Fy", "p_Fz", "visk_Fx", "visk_Fy", "visk_Fz"]


def test_read_freesurface_p() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/p_freeSurface.raw")
    assert df.shape == (278, 4)
    assert list(df.columns) == [0, 1, 2, 3]
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/p_freeSurface.raw", column_names=["x", "y", "z", "p"])
    assert df.shape == (278, 4)
    assert list(df.columns) == ["x", "y", "z", "p"]

def test_read_freesurface_u() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/U_freeSurface.raw")
    assert df.shape == (278, 6)
    assert list(df.columns) == [0, 1, 2, 3, 4, 5]
    df = reader.read("tests/test_postprocessing/postProcessing/freeSurface/0.1/U_freeSurface.raw", column_names=["x", "y", "z", "u", "v", "w"])
    assert df.shape == (278, 6)
    assert list(df.columns) == ["x", "y", "z", "u", "v", "w"]

def test_probes_p() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/p")
    assert df.shape == (109, 2)
    assert list(df.columns) == [0, 1]
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/p", column_names=["time", "p"])
    assert df.shape == (109, 2)
    assert list(df.columns) == ["time", "p"]

def test_probes_U() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/U")
    print(df)
    assert df.shape == (109, 4)
    assert list(df.columns) == [0, 1, 2, 3]
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/U", column_names=["time", "u", "v", "w"])
    assert df.shape == (109, 4)
    assert list(df.columns) == ["time", "u", "v", "w"]

def test_probes_T() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/T")
    assert df.shape == (109, 2)
    assert list(df.columns) == [0, 1]
    df = reader.read("tests/test_postprocessing/postProcessing/probes/0/T", column_names=["time", "T"])

def test_read_sample_xy() -> None:
    reader = TableReader()
    df = reader.read("tests/test_postprocessing/postProcessing/sample1/0.1/centreLine_T.xy")
    assert df.shape == (100, 2)
    assert list(df.columns) == [0, 1]
    df = reader.read("tests/test_postprocessing/postProcessing/sample1/0.1/centreLine_T.xy", column_names=["x", "y"])
    assert df.shape == (100, 2)
    assert list(df.columns) == ["x", "y"]