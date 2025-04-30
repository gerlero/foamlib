from foamlib.postprocessing.tableReader import TableReader, extract_column_names


def test_read_headers() -> None:
    force_file = "tests/test_postprocessing/postProcessing/forces/0/force.dat"
    force_headers = extract_column_names(force_file)
    assert force_headers == ["Time", "total_x", "total_y", "total_z", "pressure_x", "pressure_y", "pressure_z", "viscous_x", "viscous_y", "viscous_z"]

    surface_p_file = "tests/test_postprocessing/postProcessing/freeSurface/0.1/p_freeSurface.raw"
    surface_p_headers = extract_column_names(surface_p_file)
    assert surface_p_headers == ["x", "y", "z", "p"]

    surface_u_file = "tests/test_postprocessing/postProcessing/freeSurface/0.1/U_freeSurface.raw"
    surface_u_headers = extract_column_names(surface_u_file)
    assert surface_u_headers == ["x", "y", "z", "U_x", "U_y", "U_z"]

    probes_p_file = "tests/test_postprocessing/postProcessing/probes/0/p"
    probes_p_headers = extract_column_names(probes_p_file)
    assert probes_p_headers == ["Time", "0"]

    probes_u_file = "tests/test_postprocessing/postProcessing/probes/0/U"
    probes_u_headers = extract_column_names(probes_u_file)
    assert probes_u_headers == ["Time", "0"]

    probes_T_file = "tests/test_postprocessing/postProcessing/probes/0/T"
    probes_T_headers = extract_column_names(probes_T_file)
    assert probes_T_headers == ["Time", "0"]

    sample_xy_file = "tests/test_postprocessing/postProcessing/sample1/0.1/centreLine_T.xy"
    sample_xy_headers = extract_column_names(sample_xy_file)
    assert sample_xy_headers == None

    

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