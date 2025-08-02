#%%
import pandas as pd
import plotly.express as px

forces = pd.read_csv("results/forces.csv")
#%%
fig = px.line(
    forces,
    x="Time",
    y="total_x",
    title="Forces over Time",
    color="grid",
    symbol="initHeight",
)
fig.write_html("plot.html")
# %%
