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
hds = imod.mf6.open_hds(
    model_dir / "gwf" / "gwf.hds", model_dir / "gwf" / "dis.dis.grb"
)
hds.load()

# %% Plot
hds_selected = hds.sel(layer=1)

diff = hds_selected.isel(time=-1) - hds_selected.isel(time=0)

diff.plot.imshow()

# %%
