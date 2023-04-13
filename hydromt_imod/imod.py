# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import hydromt
import hydromt_wflow
import imod
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.models import GridModel

from . import DATADIR

__all__ = ["ImodModel"]

logger = logging.getLogger(__name__)


# create class with mesh and grid attributes.
class ImodModel(GridModel):
    _NAME = "imod"
    # mapping of names from hydromt names to model variable names
    _GEOMS = {}
    _FORCING = {}
    _MAPS = {}
    _FOLDERS = []
    _CLI_ARGS = {"region": "setup_grid"}
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

    def _raster_or_constant(
        self,
        value: Union[int, float, None],
        fn: Union[str, Path, None],
        like: xr.DataArray,
        method: str = "bilinear",
    ):
        if not ((value is None) ^ (fn is None)):
            raise ValueError(
                "Either provide a constant or a filename."
                f"constant: {value}, filename: {fn}"
            )

        elif value is not None:
            return self._prepare_constant(value, like)
        elif fn is not None:
            return self._prepare_raster(fn, like, method=method)

    def _prepare_constant(self, value, like):
        return xr.full_like(like, fill_value=value)

    def _prepare_raster(self, fn, like, method="bilinear"):
        da = self.data_catalog.get_rasterdataset(
            fn,
            geom=self.region,
            buffer=2,
            variables=fn,
            single_var_as_array=True,
        )
        return da.raster.reproject_like(like, method=method)

    # setup methods
    def setup_grid(self, region: dict, res: float, crs: int = "utm"):
        super().setup_region(region=region)
        geom = self.geoms["region"]

        if "bbox" in region.keys():
            bbox = region["bbox"]
        else:
            bbox = geom.total_bounds

        crs = hydromt.gis_utils.parse_crs(crs, bbox=bbox)

        # Force geom and region to user-defined crs for model
        # to circumvent hardcoded crs=4326 for bbox

        if crs != geom.crs:
            geom = geom.to_crs(crs)
            self.geoms["region"] = geom

        # define grid
        # calculate shape & coords
        # idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)
        # TODO mask with region?
        # set grid (coordinates only or dummy data)
        dx, dy = res, res
        xmin, ymin, xmax, ymax = geom.total_bounds
        coords = imod.util._xycoords((xmin, xmax, ymin, ymax), (dx, -dy))
        coords = {"y": coords["y"], "x": coords["x"]}
        grid_2d = hydromt.raster.full(coords, crs=geom.crs)

        self.set_grid(grid_2d, "grid_2d")

    def setup_dem(self, hydrography_fn):
        da = self.data_catalog.get_rasterdataset(
            hydrography_fn,
            geom=self.region,
            buffer=2,
            variables="elevtn",
            single_var_as_array=True,
        )
        # Reprojection
        da = da.raster.reproject_like(self.grid, method="bilinear")
        # Mask NoData
        da = da.raster.mask_nodata()
        # Add to maps
        self.set_grid(da, "dem")

    def setup_dis(self, nlay: int, bottom_fn=None, bottom_constant=None):
        bottom = self._raster_or_constant(
            bottom_constant, bottom_fn, self.grid["grid_2d"], method="bilinear"
        )

        top = self.grid["dem"]
        thickness = top - bottom
        layer_1d = xr.DataArray(
            np.ones((nlay)), coords={"layer": np.arange(nlay) + 1}, dims=("layer",)
        )
        thickness_3d = layer_1d * (thickness / nlay)
        self.set_grid(thickness_3d, "thickness")

        bottom_layers = top - thickness_3d.cumsum(dim="layer")
        bottom_layers = bottom_layers.transpose("layer", "y", "x")
        self.set_grid(bottom_layers, "bottom")

        idomain = xr.ones_like(bottom_layers, dtype=np.int32)
        idomain = idomain.where(~np.isnan(top), other=0)
        self.set_grid(idomain, "idomain")

    def setup_initial_condition(self):
        top = self.grid["dem"]
        idomain = self.grid["idomain"]

        starting_head = idomain * top
        starting_head = starting_head.where(idomain == 1)

        self.set_grid(starting_head, "starting_head")

    def setup_well(self):
        # - PCR-GLOBWB netcdf (grid)
        # - User defined, in some csv format
        pass

    def setup_storage(self, specific_yield: float, specific_storage: float):
        """

        Setup confined storage values with specific storages. To account for
        specific yield, specific yield is set in the top layer and divided by
        the top layer's thickness.

        See also: https://deltares.gitlab.io/imod/imod-python/faq/modeling.html

        """

        dz_top = self.grid["thickness"].sel(layer=1)
        specific_storage_top_layer = specific_yield / dz_top

        specific_storage_grid = xr.where(
            self.grid.coords["layer"] == 1, specific_storage_top_layer, specific_storage
        )

        specific_storage_grid = specific_storage_grid.where(self.grid["idomain"] == 1)

        self.set_grid(specific_storage_grid, "specific_storage")

    def setup_hydraulic_conductivity(
        self, hydraulic_conductivity: float, vertical_anisotropy=1.0
    ):
        # Setup K-field:
        # - GLIMPSE?
        # - Scalar value

        kh = xr.where(self.grid["idomain"], hydraulic_conductivity, np.nan)
        k33 = kh / vertical_anisotropy

        self.set_grid(kh, "k")
        self.set_grid(k33, "k33")

    def setup_overland_flow(self, resistance=1.0):
        """
        Parameters
        ----------
        resistance: float
            Resistance for overland flow.in days.
            Set to 1 day, concurrent with iMOD 5.
        """
        # Set drain to DEM level, regrid with minimum/low percentile

        elevation = self.grid["dem"].assign_coords(layer=1)
        x = self.grid.coords["x"]
        res = x[1] - x[0]
        conductance_value = res**2 / resistance
        conductance = xr.full_like(elevation, conductance_value)
        conductance = conductance.where(~np.isnan(elevation))
        ds = xr.merge([{"elevation": elevation, "conductance": conductance}])
        self.set_forcing(ds, name="overland_flow", split_dataset=False)

    def setup_recharge(
        self, rate_fn: Union[str, Path] = None, rate_constant: float = None
    ):
        # Three options possible, from complex to least complex:
        # - Recharge grid from wflow model (use full potential of hydromt)
        # - P - ET
        # - Scalar value

        rate = self._raster_or_constant(
            rate_constant, rate_fn, self.grid["grid_2d"], method="nearest"
        )

        rate = rate.assign_coords(layer=1)
        rate = rate.where(self.grid["idomain"].sel(layer=1) == 1)
        recharge = xr.Dataset(data_vars={"rate": rate})
        self.set_forcing(recharge, "recharge", split_dataset=False)

    def setup(self, region: dict, res: float, crs: int = "utm"):
        self.setup_grid(region, res, crs)
        self.setup_dem(**self.config["setup_dem"])
        self.setup_dis(**self.config["setup_dis"])
        self.setup_initial_condition()
        self.setup_storage(**self.config["setup_storage"])
        self.setup_hydraulic_conductivity(**self.config["setup_hydraulic_conductivity"])
        self.setup_recharge(**self.config["setup_recharge"])
        self.setup_overland_flow(**self.config["setup_overland_flow"])

    # I/O
    def write_grid(self):
        name = self.get_config("setup_config", "name")
        to_dir = Path(self.root) / "grid"

        gwf = imod.mf6.GroundwaterFlowModel()
        gwf["dis"] = imod.mf6.StructuredDiscretization(
            top=self.grid["dem"],
            bottom=self.grid["bottom"],
            idomain=self.grid["idomain"],
        )
        gwf["ic"] = imod.mf6.InitialConditions(start=self.grid["starting_head"])
        gwf["npf"] = imod.mf6.NodePropertyFlow(
            icelltype=0,
            k=self.grid["k"],
            k33=self.grid["k33"],
            save_flows=True,
        )

        # Create stress period which lasts 1 second to compute steady state
        # heads.
        starttime = self.get_config("setup_config", "starttime")
        coords = {"time": [starttime, starttime + pd.DateOffset(seconds=1)]}
        da_transient = xr.DataArray(data=[True, True], coords=coords, dims=("time",))

        gwf["sto"] = imod.mf6.SpecificStorage(
            specific_storage=self.grid["specific_storage"],
            specific_yield=0.0,
            transient=da_transient,
            convertible=0,
        )

        sim = imod.mf6.Modflow6Simulation(name)
        sim["gwf"] = gwf
        sim.dump(to_dir)

    def write_forcing(self):
        name = self.get_config("setup_config", "name")
        to_dir = Path(self.root) / "forcing"
        from_dir = Path(self.root) / "grid"

        fn = name + ".toml"
        sim = imod.mf6.Modflow6Simulation.from_file(from_dir / fn)

        gwf = sim["gwf"]

        if "recharge" in self.forcing.keys():
            gwf["rch"] = imod.mf6.Recharge(save_flows=True, **self.forcing["recharge"])

        if "overland_flow" in self.forcing.keys():
            gwf["olf"] = imod.mf6.Drainage(
                save_flows=True, **self.forcing["overland_flow"]
            )

        sim.dump(to_dir)

    def write_model(
        self,
    ):
        name = self.get_config("setup_config", "name")
        to_dir = Path(self.root) / "model"
        from_dir = Path(self.root) / "forcing"

        fn = name + ".toml"
        sim = imod.mf6.Modflow6Simulation.from_file(from_dir / fn)

        gwf = sim["gwf"]

        # Output Control
        oc_kwargs = dict(
            [
                (key, self.get_config("setup_config", key))
                for key in ["save_head", "save_budget"]
            ]
        )
        gwf["oc"] = imod.mf6.OutputControl(**oc_kwargs)

        # Time discretization
        additional_times = [
            self.get_config("setup_config", "starttime"),
            self.get_config("setup_config", "endtime"),
        ]
        sim.create_time_discretization(additional_times)

        # Solver
        sim["solver"] = imod.mf6.Solution(
            modelnames=["gwf"],
            print_option="summary",
            csv_output=False,
            no_ptc=True,
            outer_dvclose=1.0e-4,
            outer_maximum=500,
            under_relaxation=None,
            inner_dvclose=1.0e-4,
            inner_rclose=0.001,
            inner_maximum=100,
            linear_acceleration="cg",
            scaling_method=None,
            reordering_method=None,
            relaxation_factor=0.97,
        )

        sim.write(to_dir)

    def write(self):
        self.write_grid()
        self.write_forcing()
        self.write_model()

    ## model configuration
    def set_time(self, *args):
        time_str = self.get_config(*args)
        time = pd.to_datetime(time_str)
        self.set_config(*args, time)

    def read_config(self, config_fn: Optional[str] = None):
        super().read_config(config_fn=config_fn)
        self.set_time("setup_config", "starttime")
        self.set_time("setup_config", "endtime")
