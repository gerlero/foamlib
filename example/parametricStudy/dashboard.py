

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px


@app.cell
def _(pd):
    forces = pd.read_csv("results/forces.csv")
    return (forces,)


@app.cell
def _(forces, mo):

    x = mo.ui.dropdown(options=forces.columns, label="x",value=forces.columns[0])
    y = mo.ui.dropdown(options=forces.columns, label="y",value=forces.columns[0])
    row = mo.ui.dropdown(options=forces.columns, label="row")
    col = mo.ui.dropdown(options=forces.columns, label="col")
    color = mo.ui.dropdown(options=forces.columns, label="color")


    return col, color, row, x, y


@app.cell
def _(col, color, forces, mo, px, row, x, y):
    kwargs = {
        "x": x.value,
        "y": y.value,
        "color": None if color.value == "--" else color.value,
        # "size": None if size.value == "--" else size.value,
        # "facet_row": None if row.value == "--" else row.value,
        # "facet_col": None if col.value == "--" else col.value,
        # "symbol": None if shape.value == "--" else shape.value,
    }

    #fig_func = plot_types[mark.value]
    fig = px.line(forces, **{k: v for k, v in kwargs.items() if v is not None})
    fig.update_layout(height=600)

    mo.hstack([
        mo.vstack([mo.md("### Encodings"),x, y, row, col,mo.md("### Encodings"),color],align="end"),
        mo.ui.plotly(fig)
    ])
    return


if __name__ == "__main__":
    app.run()
