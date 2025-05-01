

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    return Path, mo, os


@app.cell
def _(Path):
    folder = Path("tests/test_postprocessing/postProcessing/freeSurface/0.2")
    return (folder,)


@app.cell
def _(folder, mo, os):
    files = os.listdir(folder)  # replace "data" with your directory
    files = [f for f in files if f.endswith(".raw")]  # filter if needed

    # Create dropdown
    file_dropdown = mo.ui.dropdown(options=files, label="Select a file")

    file_dropdown  # display dropdown
    return (file_dropdown,)


@app.cell
def _(file_dropdown, folder):
    from foamlib.postprocessing.table_reader import TableReader, extract_column_names
    force_file = folder / file_dropdown.value
    reader = TableReader()
    df = reader.read(force_file)
    df.columns = extract_column_names(filepath=force_file)
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.data_explorer(df)


if __name__ == "__main__":
    app.run()
