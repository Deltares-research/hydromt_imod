# %%
import subprocess
from pathlib import Path

import imod
import numpy as np
from hydromt.log import setuplog

from hydromt_imod.imod import ImodModel

logger = setuplog("Initialize logger", log_level=10)

root = Path(r"c:\path\to\root")
config_fn = root / r"example_imod.ini"


# %%

mod = ImodModel(root=root, mode="w", data_libs="artifact_data", logger=logger)

# Configure
mod.read_config(config_fn=config_fn)

# Tests on setup_staticmaps_from_raster
# bbox in EPSG:4326
bbox = np.array([11.37583385, 45.19166655, 13.08333353, 46.85166659])

mod.setup(region={"bbox": bbox}, res=1000.0)
mod.write()

# %% Run model
model_dir = root / "model"

# Assume Modflow 6 is added to PATH
with imod.util.cd(model_dir):
    subprocess.run("mf6 .")

# %% Read and load heads
mod.read_results()

# %% Select outputs to be plotted
hds_selected = mod.results["heads"].sel(layer=1)

start_head = hds_selected.isel(time=0)
final_head = hds_selected.isel(time=-1)
diff = final_head - start_head

# %% Plot heads
import matplotlib.pyplot as plt


def plot_data(da, title):
    fig, ax = plt.subplots()
    da.plot.imshow(ax=ax)
    ax.set_title(title)
    plt.show()


plot_data(start_head, "starting head")
plot_data(final_head, "final head")
plot_data(diff, "head difference")

# %% Plot water balance
d_labels = {
    "olf": "Dunnian runoff",
    "rch": "Recharge",
    "sto-ss_in": "Storage in",
    "sto-ss_out": "Storage out",
}

fig, ax = plt.subplots()
mod.plot_water_balance(ax, d_labels)
ax.set_title("water balance")
plt.tight_layout()
plt.show()


# %%
