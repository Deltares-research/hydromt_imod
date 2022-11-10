# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import xarray as xr
import xugrid as xu
import imod
from typing import List

import hydromt
from hydromt.models import MeshModel
from hydromt.models.model_grid import GridMixin

from . import DATADIR

__all__ = ["ImodModel"]

logger = logging.getLogger(__name__)

# create class with mesh and grid attributes.
class ImodModel(GridMixin, MeshModel):
    _NAME = "imod"
    # mapping of names from hydromt names to model variable names
    _GEOMS = {}
    _FORCING = {}
    _MAPS = {}
    _FOLDERS = []
    _CLI_ARGS = {"region": "setup_mesh"}
    _CONF = "imod.run"  # FIXME default iMOD run  (simulation configuration) file
    _DATADIR = DATADIR
    #
    _ATTRS = {}

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
    ):
        """
        The iMOD model class (ImodModel) contains methods to read, write, setup and edit
        `iMOD <https://oss.deltares.nl/web/imod>`_ models.

        Parameters
        ----------
        root: str, Path, optional
            Path to model folder
        mode: {'w', 'r+', 'r'}
            Open model in write, append or reading mode, by default 'w'
        config_fn: str, Path, optional
            Filename of model config file, by default "imod.run"
        opt: dict
            Values for model setup options, used when called from CLI
        sources: dict
            Library with references to data sources
        """
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

    # setup methods
    def setup_grid(self, region, nlay, nrow, ncol):
        super().setup_region(region=region)
        # define grid
        # calculate shape & coords
        # idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
        # TODO mask with region?
        # set grid (coordinates only or dummy data)
        # self.set_grid(idomain, 'idomain')

    def setup_dis(self, top_fn, bottom_fn):
        # NOTE: perhaps this and the above setup_mesh method can be combined
        # read data
        # top = self.data_catalog.get_rasterdataset(top_fn, geom=self.region, buffer=5,)
        # bottom = self.data_catalog.get_rasterdataset(bottom_fn, geom=self.region, buffer=5,)
        # resample to grid ??
        # ds_dis = imod.mf6.StructuredDiscretization(
        #     top=top, bottom=bottom, idomain=self.grid['idomain']
        # )
        # add data to grid
        # self.set_grid(ds_dis, 'dis')
        pass

    def setup_recharge(self):
        # # read data
        # da_recharge = self.data_catalog.get_rasterdataset(recharge_fn, geom=self.region, buffer=5,)

        # # transform data
        # da_recharge = imod.regrid(da_recharge)

        # # add data to forcing attribute
        # self.set_forcing(da_recharge, 'recharge')
        pass

    # I/O
    def write_grid(self, fn: str = "grid/grid.nc", **kwargs):
        super().write_grid(fn=fn, **kwargs)

    def write_mesh(self, *args, **kwargs):
        # NOTE this code comes from the MeshMODEL class, but can be overwritten here
        # if self._mesh is None:
        #     self.logger.debug("No mesh data found, skip writing.")
        #     return
        # self._assert_write_mode
        # # filename
        # _fn = join(self.root, fn)
        # if not isdir(dirname(_fn)):
        #     os.makedirs(dirname(_fn))
        # self.logger.debug(f"Writing file {fn}")
        # # ds_new = xu.UgridDataset(grid=ds_out.ugrid.grid) # bug in xugrid?
        # ds_out = self.mesh.ugrid.to_dataset()
        # if self.mesh.ugrid.grid.crs is not None:
        #     # save crs to spatial_ref coordinate
        #     ds_out = ds_out.rio.write_crs(self.mesh.ugrid.grid.crs)
        # ds_out.to_netcdf(_fn, **kwargs)
        pass

    ## model configuration

    def _configread(self, fn):
        # FIXME if default ini reader does not work
        # return imod.read_config(fn)
        pass

    def _configwrite(self, fn):
        # FIXME
        # return imod.write_config(fn, self.config)
        pass
