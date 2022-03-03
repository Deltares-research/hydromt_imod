# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import xarray as xr
import imod

import hydromt
from hydromt.models.model_api import Model

from . import DATADIR

__all__ = ["ImodModel"]

logger = logging.getLogger(__name__)


class ImodModel(Model):
    _NAME = "imod"
    # mapping of names from hydromt names to model variable names
    _GEOMS = {}  
    _FORCING = {}  
    _MAPS = {} 
    _FOLDERS = []
    _CLI_ARGS = {"region": "setup_region", "res": "setup_basemaps"}
    _CONF = "imod.run"  # FIXME default iMOD run  (simulation configuration) file
    _DATADIR = DATADIR
    # 
    _ATTRS = {}

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn="imod.run",
        data_libs=None,
        deltares_data=None,
        artifact_data=None,
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
            deltares_data=deltares_data,
            artifact_data=artifact_data,
            logger=logger,
        )

    def setup_basemaps(self):
        # FIXME
        pass

    # I/O
    def read(self):
        """Read the complete model schematization and configuration from file."""
        # FIXME: remove unused functions
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Write the complete model schematization and configuration to file."""
        # FIXME: remove unused functions
        self.logger.info(f"Writing model data to {self.root}")
        self.write_staticmaps()
        self.write_staticgeoms()
        self.write_forcing()
        self.write_states()
        # config last; might be updated when writing maps, states or forcing
        self.write_config()
        # write data catalog with used data sources
        # self.write_data_catalog()  # new in hydromt v0.4.4

    def read_staticmaps(self):
        """Read iMOD staticmaps and save to `staticmaps` attribute (xarray.Dataset).
        """
        # FIXME
        # ds = imod.read()
        # self.set_staticmaps(ds)


    def write_staticmaps(self):
        """Write iMOD staticmaps to model files."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # FIXME
        # imod.write_mode(self.staticmaps)


    def read_staticgeoms(self):
        """Read geometry (vector) files and save to `staticgeoms` attribute (dictionary of geopandas.GeoDataFrame).
        """
        # For any 1D schematization, not sure this is applicable to iMOD
        # FIXME
        # gdf = imod.read_
        # self.set_staticgeom(gdf, name='')


    def write_staticgeoms(self):
        """Write staticgeoms to model files
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        pass

    def read_forcing(self):
        """Read forcing files and save to `forcing` attribute (dictionary of xarray.DataArray).
        """
        # FIXME
        # da = imod.read_recharge()
        # self.set_forcing(da, name='recharge')
        # imod.read_head()
        # self.set_forcing(da, name='head')

    def write_forcing(self):
        """Write forcing to model files.
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # FIXME
        # imod.write_recharge()
        # imod.write_head()

    def read_states(self, crs=None):
        """Read imod state files and save to `states` attribute (dictionary of xarray.DataArray).
        """
        # FIXME
        # self.set_state()

    def write_states(self):
        """Write imod state to model files
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # FIXME

    def read_results(
        self,
    ):
        """Read imod model results and save to `results` attribute (dictionary of xarray.DataArray).
        """
        # ds_results = 
        # self.set_results(ds_results, split_dataset=True)

    ## model configuration

    def set_crs(self, crs):
        super(ImodModel, self).set_crs(crs)
        self.update_spatial_attrs()

    def _configread(self, fn):
        # FIXME if default ini reader does not work
        # return imod.read_config(fn)
        pass

    def _configwrite(self, fn):
        # FIXME
        # return imod.write_config(fn, self.config)
        pass