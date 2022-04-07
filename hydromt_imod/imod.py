"""Implement wflow model class"""
# Implement model class following model API

import os
from os.path import join, dirname, basename, isfile, isdir
from typing import Optional
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import pyproj
import toml
import codecs
from pyflwdir import core_d8, core_ldd, core_conversion
from dask.diagnostics import ProgressBar
import logging
import imod

# from dask.distributed import LocalCluster, Client, performance_report
import hydromt
from hydromt.models.model_api import Model
from hydromt import flw
from hydromt.io import open_mfraster

from . import utils, workflows, DATADIR


__all__ = ["ImodModel"]

logger = logging.getLogger(__name__)

# specify pcraster map types
# NOTE non scalar (float) data types only
PCR_VS_MAP = {
    "ldd": "ldd",
    "river": "bool",
    "streamorder": "ordinal",
    "gauges": "nominal",  # to avoid large memory usage in pcraster.aguila
    "basins": "nominal",  # idem.
    "landuse": "nominal",
    "soil": "nominal",
    "reservoirareas": "nominal",
    "reservoirlocs": "nominal",
    "lakeareas": "nominal",
    "lakelocs": "nominal",
    "top_1": "top_1",
}


class ImodModel(Model):
    """This is the imod model class"""

    _NAME = "imod"
    _CONF = "imod.run"
    _DATADIR = DATADIR
    _GEOMS = {}
    _MAPS = {
        "flwdir": "ldd",
        "strord": "riv_1",        
        "bnd_1": "bnd_1",
        "rivlen": "riverlength",
        "rivmsk": "river",
        "rivwth": "riverwidth",
        "rivslp": "riverslope",
        "lndslp": "Slope",
        "rivman": "N_River",
        "gauges": "gauges",
        "landuse": "landuse",
        "resareas": "res_1",
        "reslocs": "reservoirlocs",
        "lakeareas": "lak_1",
        "lakelocs": "lakelocs",
        "uparea": "uparea",
    }
    _FOLDERS = [
        "staticgeoms",
    ]

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,
        data_libs=None,
        deltares_data=False,
        logger=logger,
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            deltares_data=deltares_data,
            logger=logger,
        )

        # wflow specific
        self._intbl = dict()
        self._flwdir = None

    # COMPONENTS
    def setup_basemaps(
        self,
        region,
        res=1 / 120.0,
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        geology_fn="soilgrids", 
        ptf_ksatver="brakensiek",
        upscale_method="ihu",
    ):
        """
        This component sets the ``region`` of interest and ``res`` (resolution in degrees) of the
        model. All DEM and flow direction related maps are then build.

        If the model resolution is larger than the source data resolution,
        the flow direction is upscaled using the ``upscale_method``, by default the
        Iterative Hydrography Upscaling (IHU).
        The default ``hydrography_fn`` is "merit_hydro" (`MERIT hydro <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/index.html>`_
        at 3 arcsec resolution) Alternative sources include "merit_hydro_1k" at 30 arcsec resolution.
        Users can also supply their own elevation and flow direction data. Note that only EPSG:4326 base data supported.

        Adds model layers:

        * **ldd** map: flow direction in LDD format [-]
        * **basins** map: basin ID map [-]
        * **uparea** map: upstream area [km2]
        * **streamorder** map: Strahler stream order [-]
        * **dem** map: average elevation [m+REF]
        * **dem_subgrid** map: subgrid outlet elevation [m+REF]
        * **Slope** map: average land surface slope [m/m]
        * **basins** geom: basins boundary vector
        * **region** geom: region boundary vector

        Parameters
        ----------
        hydrography_fn : str
            Name of data source for basemap parameters.

            * Required variables: ['flwdir', 'uparea', 'basins', 'strord', 'top_1']

            * Optional variables: ['lndslp', 'mask']
        basin_index_fn : str
            Name of data source for basin_index data linked to hydrography_fn.
        region : dict
            Dictionary describing region of interest.
            See :py:function:~basin_mask.parse_region for all options
        res : float
            Output model resolution
        upscale_method : {'ihu', 'eam', 'dmm'}
            Upscaling method for flow direction data, by default 'ihu'.

        See Also
        --------
        hydromt.workflows.parse_region
        hydromt.workflows.get_basin_geometry
        workflows.hydrography
        workflows.topography
        """
        self.logger.info(f"Preparing base hydrography basemaps.")
        # retrieve global data (lazy!)
        ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
        # TODO support and test (user) data from other sources with other crs!
        if ds_org.raster.crs is None or ds_org.raster.crs.to_epsg() != 4326:
            raise ValueError("Only EPSG:4326 base data supported.")
        # get basin geometry and clip data
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        xy = None
        if kind in ["basin", "subbasin", "outlet"]:
            bas_index = self.data_catalog[basin_index_fn]
            geom, xy = hydromt.workflows.get_basin_geometry(
                ds=ds_org,
                kind=kind,
                basin_index=bas_index,
                logger=self.logger,
                **region,
            )
        elif "bbox" in region:
            bbox = region.get("bbox")
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif "geom" in region:
            geom = region.get("geom")
        else:
            raise ValueError(f"wflow region argument not understood: {region}")
        if geom is not None and geom.crs is None:
            raise ValueError("wflow region geometry has no CRS")
        ds_org = ds_org.raster.clip_geom(geom, align=res, buffer=10)
        self.logger.debug(f"Adding basins vector to staticgeoms.")
        self.set_staticgeoms(geom, name="basins")

        # setup hydrography maps and set staticmap attribute with renamed maps
        ds_base = workflows.hydrography(
            geology_fn,
            ds=ds_org,
            res=res,
            xy=xy,
            upscale_method=upscale_method,
            logger=self.logger,
        )
        # Convert flow direction from d8 to ldd format
        #flwdir_data = ds_base["flwdir"].values.astype(np.uint8)  # force dtype
        # if d8 convert to ldd
        #if core_d8.isvalid(flwdir_data):
        #    data = core_conversion.d8_to_ldd(flwdir_data)
        #    da_flwdir = xr.DataArray(
        #        name="flwdir",
        #        data=data,
        #        coords=ds_base.raster.coords,
        #        dims=ds_base.raster.dims,
        #        attrs=dict(
        #            long_name="ldd flow direction",
        #            _FillValue=core_ldd._mv,
        #        ),
        #    )
        #    ds_base["flwdir"] = da_flwdir
        # Rename and add to staticmaps
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_base.data_vars}       
        self.set_staticmaps(ds_base.rename(rmdict))

        # setup topography maps
        ds_topo = workflows.topography(
            geology_fn, ds=ds_org, ds_like=self.staticmaps, method="average", logger=self.logger
        )
                       
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_topo.data_vars}
        self.set_staticmaps(ds_topo.rename(rmdict))
        
        # set basin geometry
        self.logger.debug(f"Adding region vector to staticgeoms.")
        self.set_staticgeoms(self.region, name="region")
        
        self.logger.info(f"Preparing soil parameter maps.")
        # TODO add variables list with required variable names
        dsin = self.data_catalog.get_rasterdataset(geology_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids(
            dsin,
            self.staticmaps,
            ptf_ksatver,
            geology_fn,
            logger=self.logger,
        )
        self.set_staticmaps(dsout)

   
    def setup_gauges(
        self,
        gauges_fn="grdc",
        source_gdf=None,
        index_col=None,
        snap_to_river=True,
        mask=None,
        derive_subcatch=False,
        derive_outlet=True,
        basename=None,
        update_toml=True,
        gauge_toml_header=None,
        gauge_toml_param=None,
        **kwargs,
    ):
        """This components sets the default gauge map based on basin outlets and additional
        gauge maps based on ``gauges_fn`` data.

        Supported gauge datasets include "grdc"
        or "<path_to_source>" for user supplied csv or geometry files with gauge locations.
        If a csv file is provided, a "x" or "lon" and "y" or "lat" column is required
        and the first column will be used as IDs in the map. If ``snap_to_river`` is set
        to True, the gauge location will be snapped to the boolean river mask. If
        ``derive_subcatch`` is set to True, an additional subcatch map is derived from
        the gauge locations.

        Adds model layers:

        * **gauges** map: gauge IDs map from catchment outlets [-]
        * **gauges_source** map: gauge IDs map from source [-] (if gauges_fn)
        * **wflow_subcatch_source** map: subcatchment based on gauge locations [-] (if derive_subcatch)
        * **gauges** geom: polygon of catchment outlets
        * **gauges_source** geom: polygon of gauges from source
        * **subcatch_source** geom: polygon of subcatchment based on gauge locations [-] (if derive_subcatch)

        Parameters
        ----------
        gauges_fn : str, {"grdc"}, optional
            Known source name or path to gauges file geometry file, by default None.
        source_gdf : geopandas.GeoDataFame, optional
            Direct gauges file geometry, by default None.
        index_col : str, optional
            Column in gauges_fn to use for ID values, by default None (use the default index column)
        snap_to_river : bool, optional
            Snap point locations to the closest downstream river cell, by default True
        mask : np.boolean, optional
            If provided snaps to the mask, else snaps to the river (default).
        derive_subcatch : bool, optional
            Derive subcatch map for gauges, by default False
        derive_outlet : bool, optional
            Derive gaugemap based on catchment outlets, by default True
        basename : str, optional
            Map name in staticmaps (gauges_basename), if None use the gauges_fn basename.
        update_toml : boolean, optional
            Update [outputcsv] section of wflow toml file.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section. This option defines the header of the csv file./
            By default saves Q (for lateral.river.q_av) and P (for vertical.precipitation).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines the wflow variable corresponding to the/
            names in gauge_toml_header. By default saves lateral.river.q_av (for Q) and vertical.precipitation (for P).
        """
        # read existing staticgeoms; important to get the right basin when updating
        self.staticgeoms

        if derive_outlet:
            self.logger.info(f"Gauges locations set based on river outlets.")
            da, idxs, ids = flw.gauge_map(self.staticmaps, idxs=self.flwdir.idxs_pit)
            # Only keep river outlets for gauges
            da = da.where(self.staticmaps[self._MAPS["rivmsk"]], da.raster.nodata)
            ids_da = np.unique(da.values[da.values > 0])
            idxs_da = idxs[np.isin(ids, ids_da)]
            self.set_staticmaps(da, name=self._MAPS["gauges"])
            points = gpd.points_from_xy(*self.staticmaps.raster.idx_to_xy(idxs_da))
            gdf = gpd.GeoDataFrame(
                index=ids_da.astype(np.int32), geometry=points, crs=self.crs
            )
            gdf["fid"] = ids_da.astype(np.int32)
            self.set_staticgeoms(gdf, name="gauges")
            self.logger.info(f"Gauges map based on catchment river outlets added.")

        if gauges_fn is not None or source_gdf is not None:
            # append location from geometry
            # TODO check snapping locations based on upstream area attribute of the gauge data
            if gauges_fn is not None:
                kwargs = {}
                if isfile(gauges_fn):
                    # try to get epsg number directly, important when writting back data_catalog
                    if self.crs.is_epsg_code:
                        code = int(self.crs["init"].lstrip("epsg:"))
                    else:
                        code = self.crs
                    kwargs.update(crs=code)
                gdf = self.data_catalog.get_geodataframe(
                    gauges_fn, geom=self.basins, assert_gtype="Point", **kwargs
                )
                gdf = gdf.to_crs(self.crs)
            elif source_gdf is not None and basename is None:
                raise ValueError(
                    "Basename is required when setting gauges based on source_gdf"
                )
            elif source_gdf is not None:
                self.logger.info(f"Gauges locations read from source_gdf")
                gdf = source_gdf.to_crs(self.crs)

            if gdf.index.size == 0:
                self.logger.warning(
                    f"No {gauges_fn} gauge locations found within domain"
                )
            else:
                if basename is None:
                    basename = (
                        os.path.basename(gauges_fn).split(".")[0].replace("_", "-")
                    )
                self.logger.info(
                    f"{gdf.index.size} {basename} gauge locations found within domain"
                )
                # Set index to index_col
                if index_col is not None and index_col in gdf:
                    gdf = gdf.set_index(index_col)
                xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(
                    gdf["geometry"]
                )
                idxs = self.staticmaps.raster.xy_to_idx(xs, ys)
                ids = gdf.index.values

                if snap_to_river and mask is None:
                    mask = self.staticmaps[self._MAPS["rivmsk"]].values
                da, idxs, ids = flw.gauge_map(
                    self.staticmaps,
                    idxs=idxs,
                    ids=ids,
                    stream=mask,
                    flwdir=self.flwdir,
                    logger=self.logger,
                )
                # Filter gauges that could not be snapped to rivers
                if snap_to_river:
                    ids_old = ids.copy()
                    da = da.where(
                        self.staticmaps[self._MAPS["rivmsk"]], da.raster.nodata
                    )
                    ids_new = np.unique(da.values[da.values > 0])
                    idxs = idxs[np.isin(ids_old, ids_new)]
                    ids = da.values.flat[idxs]
                # Add to staticmaps
                mapname = f'{str(self._MAPS["gauges"])}_{basename}'
                self.set_staticmaps(da, name=mapname)

                # geoms
                points = gpd.points_from_xy(*self.staticmaps.raster.idx_to_xy(idxs))
                # if csv contains additional columns, these are also written in the staticgeoms
                gdf_snapped = gpd.GeoDataFrame(
                    index=ids.astype(np.int32), geometry=points, crs=self.crs
                )
                # Set the index name of gdf snapped based on original gdf
                if gdf.index.name is not None:
                    gdf_snapped.index.name = gdf.index.name
                else:
                    gdf_snapped.index.name = "fid"
                    gdf.index.name = "fid"
                # Add gdf attributes to gdf_snapped (filter on snapped index before merging)
                df_attrs = pd.DataFrame(gdf.drop(columns="geometry"))
                df_attrs = df_attrs[np.isin(df_attrs.index, gdf_snapped.index)]
                gdf_snapped = gdf_snapped.merge(
                    df_attrs, how="inner", on=gdf.index.name
                )
                # Add gdf_snapped to staticgeoms
                self.set_staticgeoms(gdf_snapped, name=mapname.replace("", ""))

                # # Add new outputcsv section in the config
                if gauge_toml_param is None and update_toml:
                    gauge_toml_header = ["Q", "P"]
                    gauge_toml_param = ["lateral.river.q_av", "vertical.precipitation"]

                if update_toml:
                    self.set_config(f"input.gauges_{basename}", f"{mapname}")
                    if self.get_config("csv") is not None:
                        for o in range(len(gauge_toml_param)):
                            gauge_toml_dict = {
                                "header": gauge_toml_header[o],
                                "map": f"gauges_{basename}",
                                "parameter": gauge_toml_param[o],
                            }
                            # If the gauge outcsv column already exists skip writting twice
                            if gauge_toml_dict not in self.config["csv"]["column"]:
                                self.config["csv"]["column"].append(gauge_toml_dict)
                    self.logger.info(f"Gauges map from {basename} added.")

                # add subcatch
                if derive_subcatch:
                    da_basins = flw.basin_map(
                        self.staticmaps, self.flwdir, idxs=idxs, ids=ids
                    )[0]
                    mapname = self._MAPS["bnd_1"] + "_" + basename
                    self.set_staticmaps(da_basins, name=mapname)
                    gdf_basins = flw.basin_shape(
                        self.staticmaps, self.flwdir, basin_name=mapname
                    )
                    self.set_staticgeoms(gdf_basins, name=mapname.replace("", ""))

    
    def setup_lakes(self, lakes_fn="hydro_lakes", min_area=10.0):
        """This component generates maps of lake areas and outlets as well as parameters
        with average lake area, depth a discharge values.

        The data is generated from features with ``min_area`` [km2] (default 1 km2) from a database with
        lake geometry, IDs and metadata. Data required are lake ID 'waterbody_id', average area 'Area_avg' [m2],
        average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m] and average discharge 'Dis_avg' [m3/s].

        Adds model layers:

        * **lakeareas** map: lake IDs [-]
        * **lakelocs** map: lake IDs at outlet locations [-]
        * **LakeArea** map: lake area [m2]
        * **LakeAvgLevel** map: lake average water level [m]
        * **LakeAvgOut** map: lake average discharge [m3/s]
        * **Lake_b** map: lake rating curve coefficient [-]
        * **lakes** geom: polygon with lakes and wflow lake parameters

        Parameters
        ----------
        lakes_fn : {'hydro_lakes'}
            Name of data source for lake parameters, see data/data_sources.yml.

            * Required variables: ['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg']
        min_area : float, optional
            Minimum lake area threshold [km2], by default 1.0 km2.
        """

        lakes_toml = {
            "model.lakes": True,
            "state.lateral.river.lake.waterlevel": "waterlevel_lake",
            "input.lateral.river.lake.area": "LakeArea",
            "input.lateral.river.lake.areas": "lakeareas",
            "input.lateral.river.lake.b": "Lake_b",
            "input.lateral.river.lake.e": "Lake_e",
            "input.lateral.river.lake.locs": "lakelocs",
            "input.lateral.river.lake.outflowfunc": "LakeOutflowFunc",
            "input.lateral.river.lake.storfunc": "LakeStorFunc",
            "input.lateral.river.lake.threshold": "LakeThreshold",
            "input.lateral.river.lake.linkedlakelocs": "LinkedLakeLocs",
            "input.lateral.river.lake.waterlevel": "LakeAvgLevel",
        }

        gdf_org, ds_lakes = self._setup_waterbodies(lakes_fn, "lake", min_area)
        if ds_lakes is not None:
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lakes.data_vars}
            self.set_staticmaps(ds_lakes.rename(rmdict))
            # add waterbody parameters
            # rename to param values
            gdf_org = gdf_org.rename(
                columns={
                    "Area_avg": "LakeArea",
                    "Depth_avg": "LakeAvgLevel",
                    "Dis_avg": "LakeAvgOut",
                }
            )
            # Minimum value for LakeAvgOut
            LakeAvgOut = gdf_org["LakeAvgOut"].copy()
            gdf_org["LakeAvgOut"] = np.maximum(gdf_org["LakeAvgOut"], 0.01)
            gdf_org["Lake_b"] = gdf_org["LakeAvgOut"].values / (
                gdf_org["LakeAvgLevel"].values
            ) ** (2)
            gdf_org["Lake_e"] = 2
            gdf_org["LakeStorFunc"] = 1
            gdf_org["LakeOutflowFunc"] = 3
            gdf_org["LakeThreshold"] = 0.0
            gdf_org["LinkedLakeLocs"] = 0

            # Check if some LakeAvgOut values have been replaced
            if not np.all(LakeAvgOut == gdf_org["LakeAvgOut"]):
                self.logger.warning(
                    "Some values of LakeAvgOut have been replaced by a minimum value of 0.01m3/s"
                )

            lake_params = [
                "waterbody_id",
                "LakeArea",
                "LakeAvgLevel",
                "LakeAvgOut",
                "Lake_b",
                "Lake_e",
                "LakeStorFunc",
                "LakeOutflowFunc",
                "LakeThreshold",
                "LinkedLakeLocs",
            ]

            gdf_org_points = gpd.GeoDataFrame(
                gdf_org[lake_params],
                geometry=gpd.points_from_xy(gdf_org.xout, gdf_org.yout),
            )

            # write lakes with attr tables to static geoms.
            self.set_staticgeoms(gdf_org, name="lakes")

            for name in lake_params[1:]:
                da_lake = ds_lakes.raster.rasterize(
                    gdf_org_points, col_name=name, dtype="float32", nodata=-999
                )
                self.set_staticmaps(da_lake)

            # if there are lakes, change True in toml
            for option in lakes_toml:
                self.set_config(option, lakes_toml[option])

    def setup_reservoirs(
        self,
        reservoirs_fn="hydro_reservoirs",
        min_area=1.0,
        priority_jrc=True,
        **kwargs,
    ):
        """This component generates maps of reservoir areas and outlets as well as parameters
        with average reservoir area, demand, min and max target storage capacities and
        discharge capacity values.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Data requirements for direct use (ie wflow parameters are data already present in reservoirs_fn)
        are reservoir ID 'waterbody_id', area 'ResSimpleArea' [m2], maximum volume 'ResMaxVolume' [m3],
        the targeted minimum and maximum fraction of water volume in the reservoir 'ResTargetMinFrac'
        and 'ResTargetMaxFrac' [-], the average water demand ResDemand [m3/s] and the maximum release of
        the reservoir before spilling 'ResMaxRelease' [m3/s].

        In case the wflow parameters are not directly available they can be computed by HydroMT using other
        reservoir characteristics. If not enough characteristics are available, the hydroengine tool will be
        used to download additionnal timeseries from the JRC database.
        The required variables for computation of the parameters with hydroengine are reservoir ID 'waterbody_id',
        reservoir ID in the HydroLAKES database 'Hylak_id', average volume 'Vol_avg' [m3], average depth 'Depth_avg'
        [m], average discharge 'Dis_avg' [m3/s] and dam height 'Dam_height' [m].
        To compute parameters without using hydroengine, the required varibales in reservoirs_fn are reservoir ID 'waterbody_id',
        average area 'Area_avg' [m2], average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m], average discharge 'Dis_avg'
        [m3/s] and dam height 'Dam_height' [m] and minimum / normal / maximum storage capacity of the dam 'Capacity_min',
        'Capacity_norm', 'Capacity_max' [m3].



        Adds model layers:

        * **reservoirareas** map: reservoir IDs [-]
        * **reservoirlocs** map: reservoir IDs at outlet locations [-]
        * **ResSimpleArea** map: reservoir area [m2]
        * **ResMaxVolume** map: reservoir max volume [m3]
        * **ResTargetMinFrac** map: reservoir target min frac [m3/m3]
        * **ResTargetFullFrac** map: reservoir target full frac [m3/m3]
        * **ResDemand** map: reservoir demand flow [m3/s]
        * **ResMaxRelease** map: reservoir max release flow [m3/s]
        * **reservoirs** geom: polygon with reservoirs and wflow reservoir parameters

        Parameters
        ----------
        reservoirs_fn : {'hydro_reservoirs'}
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables for direct use: ['waterbody_id', 'ResSimpleArea', 'ResMaxVolume', 'ResTargetMinFrac', 'ResTargetFullFrac', 'ResDemand', 'ResMaxRelease']

            * Required variables for computation with hydroengine: ['waterbody_id', 'Hylak_id', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Dam_height']

            * Required variables for computation without hydroengine: ['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Capacity_max', 'Capacity_norm', 'Capacity_min', 'Dam_height']
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.
        priority_jrc : boolean, optional
            If True, use JRC water occurrence (Pekel,2016) data from GEE to calculate
            and overwrite the reservoir volume/areas of the data source.
        """
        # rename to wflow naming convention
        tbls = {
            "resarea": "ResSimpleArea",
            "resdemand": "ResDemand",
            "resfullfrac": "ResTargetFullFrac",
            "resminfrac": "ResTargetMinFrac",
            "resmaxrelease": "ResMaxRelease",
            "resmaxvolume": "ResMaxVolume",
            "resid": "expr1",
        }

        res_toml = {
            "model.reservoirs": True,
            "state.lateral.river.reservoir.volume": "volume_reservoir",
            "input.lateral.river.reservoir.area": "ResSimpleArea",
            "input.lateral.river.reservoir.areas": "reservoirareas",
            "input.lateral.river.reservoir.demand": "ResDemand",
            "input.lateral.river.reservoir.locs": "reservoirlocs",
            "input.lateral.river.reservoir.maxrelease": "ResMaxRelease",
            "input.lateral.river.reservoir.maxvolume": "ResMaxVolume",
            "input.lateral.river.reservoir.targetfullfrac": "ResTargetFullFrac",
            "input.lateral.river.reservoir.targetminfrac": "ResTargetMinFrac",
        }

        gdf_org, ds_res = self._setup_waterbodies(reservoirs_fn, "reservoir", min_area)
        # TODO: check if there are missing values in the above columns of the parameters tbls =
        # if everything is present, skip calculate reservoirattrs() and directly make the maps
        if ds_res is not None:
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_res.data_vars}
            self.set_staticmaps(ds_res.rename(rmdict))

            # add attributes
            # if present use directly
            resattributes = [
                "waterbody_id",
                "ResSimpleArea",
                "ResMaxVolume",
                "ResTargetMinFrac",
                "ResTargetFullFrac",
                "ResDemand",
                "ResMaxRelease",
            ]
            if np.all(np.isin(resattributes, gdf_org.columns)):
                intbl_reservoirs = gdf_org[resattributes]
                reservoir_accuracy = None
            # else compute
            else:
                intbl_reservoirs, reservoir_accuracy = workflows.reservoirattrs(
                    gdf=gdf_org,
                    priorityJRC=priority_jrc,
                    usehe=kwargs.get("usehe", True),
                    logger=self.logger,
                )
                intbl_reservoirs = intbl_reservoirs.rename(columns=tbls)

            # create a geodf with id of reservoir and gemoetry at outflow location
            gdf_org_points = gpd.GeoDataFrame(
                gdf_org["waterbody_id"],
                geometry=gpd.points_from_xy(gdf_org.xout, gdf_org.yout),
            )
            intbl_reservoirs = intbl_reservoirs.rename(
                columns={"expr1": "waterbody_id"}
            )
            gdf_org_points = gdf_org_points.merge(
                intbl_reservoirs, on="waterbody_id"
            )  # merge
            # add parameter attributes to polygone gdf:
            gdf_org = gdf_org.merge(intbl_reservoirs, on="waterbody_id")

            # write reservoirs with param values to staticgeoms
            self.set_staticgeoms(gdf_org, name="reservoirs")

            for name in gdf_org_points.columns[2:]:
                gdf_org_points[name] = gdf_org_points[name].astype("float32")
                da_res = ds_res.raster.rasterize(
                    gdf_org_points, col_name=name, dtype="float32", nodata=-999
                )
                self.set_staticmaps(da_res)

            # Save accuracy information on reservoir parameters
            if reservoir_accuracy is not None:
                reservoir_accuracy.to_csv(join(self.root, "reservoir_accuracy.csv"))

            for option in res_toml:
                self.set_config(option, res_toml[option])

    def _setup_waterbodies(self, waterbodies_fn, wb_type, min_area=0.0):
        """Helper method with the common workflow of setup_lakes and setup_reservoir.
        See specific methods for more info about the arguments."""
        # retrieve data for basin
        self.logger.info(f"Preparing {wb_type} maps.")
        gdf_org = self.data_catalog.get_geodataframe(
            waterbodies_fn, geom=self.basins, predicate="contains"
        )
        # skip small size waterbodies
        if "Area_avg" in gdf_org.columns and gdf_org.geometry.size > 0:
            min_area_m2 = min_area * 1e6
            gdf_org = gdf_org[gdf_org.Area_avg >= min_area_m2]
        else:
            self.logger.warning(
                f"{wb_type}'s database has no area attribute. "
                f"All {wb_type}s will be considered."
            )
        # get waterbodies maps and parameters
        nb_wb = gdf_org.geometry.size
        ds_waterbody = None
        if nb_wb > 0:
            self.logger.info(
                f"{nb_wb} {wb_type}(s) of sufficient size found within region."
            )
            # add waterbody maps
            uparea_name = self._MAPS["uparea"]
            if uparea_name not in self.staticmaps.data_vars:
                self.logger.warning(
                    f"Upstream area map for {wb_type} outlet setup not found. "
                    "Database coordinates used instead"
                )
                uparea_name = None
            ds_waterbody, gdf_wateroutlet = workflows.waterbodymaps(
                gdf=gdf_org,
                ds_like=self.staticmaps,
                wb_type=wb_type,
                uparea_name=uparea_name,
                logger=self.logger,
            )
            # update xout and yout in gdf_org from gdf_wateroutlet:
            if "xout" in gdf_org.columns and "yout" in gdf_org.columns:
                gdf_org.loc[:, "xout"] = gdf_wateroutlet["xout"]
                gdf_org.loc[:, "yout"] = gdf_wateroutlet["yout"]

        else:
            self.logger.warning(
                f"No {wb_type}s of sufficient size found within region! "
                f"Skipping {wb_type} procedures!"
            )

        # rasterize points polygons in raster.rasterize -- you need staticmaps to nkow the grid
        return gdf_org, ds_waterbody

    
    def setup_constant_pars(self, **kwargs):
        """Setup constant parameter maps.

        Adds model layer:

        * **param_name** map: constant parameter map.

        Parameters
        ----------
        name = value : list of opt
            Will add name in staticmaps with value for every active cell.

        """
        for key, value in kwargs.items():
            nodatafloat = -999
            da_param = xr.where(
                self.staticmaps[self._MAPS["bnd_1"]], value, nodatafloat
            )
            da_param.raster.set_nodata(nodatafloat)

            da_param = da_param.rename(key)
            self.set_staticmaps(da_param)

    def setup_precip_forcing(
        self,
        precip_fn: str = "era5",
        precip_clim_fn: Optional[str] = None,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Setup gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5
            Precipitation data source, see data/forcing_sources.yml.

            * Required variable: ['precip']
        precip_clim_fn : str, default None
            High resolution climatology precipitation data source to correct precipitation,
            see data/forcing_sources.yml.

            * Required variable: ['precip']
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if precip_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")
        mask = self.staticmaps[self._MAPS["bnd_1"]].values > 0

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )

        if chunksize is not None:
            precip = precip.chunk({"time": chunksize})

        clim = None
        if precip_clim_fn != None:
            clim = self.data_catalog.get_rasterdataset(
                precip_clim_fn,
                geom=precip.raster.box,
                buffer=2,
                variables=["precip"],
            )

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.staticmaps[self._MAPS["bnd_1"]],
            clim=clim,
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn})
        if precip_clim_fn is not None:
            precip_out.attrs.update({"precip_clim_fn": precip_clim_fn})
        self.set_forcing(precip_out.where(mask), name="precip")

    def setup_temp_pet_forcing(
        self,
        temp_pet_fn: str = "era5",
        pet_method: str = "debruin",
        press_correction: bool = True,
        temp_correction: bool = True,
        dem_forcing_fn: str = "era5_orography",
        skip_pet: str = False,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Setup gridded reference evapotranspiration forcing at model resolution.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]
        * **temp**: temperature [°C]

        Parameters
        ----------
        temp_pet_fn : str, optional
            Name or path of data source with variables to calculate temperature and reference evapotranspiration,
            see data/forcing_sources.yml, by default 'era5'.

            * Required variable for temperature: ['temp']

            * Required variables for De Bruin reference evapotranspiration: ['temp', 'press_msl', 'kin', 'kout']

            * Required variables for Makkink reference evapotranspiration: ['temp', 'press_msl', 'kin']
        pet_method : {'debruin', 'makkink'}, optional
            Reference evapotranspiration method, by default 'debruin'.
        press_correction, temp_correction : bool, optional
             If True pressure, temperature are corrected using elevation lapse rate,
             by default False.
        dem_forcing_fn : str, default None
             Elevation data source with coverage of entire meteorological forcing domain.
             If temp_correction is True and dem_forcing_fn is provided this is used in
             combination with elevation at model resolution to correct the temperature.
        skip_pet : bool, optional
            If True calculate temp only.
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if temp_pet_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        timestep = self.get_config("timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        mask = self.staticmaps[self._MAPS["bnd_1"]].values > 0

        variables = ["temp"]
        if not skip_pet:
            if pet_method == "debruin":
                variables += ["press_msl", "kin", "kout"]
            elif pet_method == "makkink":
                variables += ["press_msl", "kin"]
            else:
                methods = ["debruin", "makking"]
                ValueError(f"Unknown pet method {pet_method}, select from {methods}")

        ds = self.data_catalog.get_rasterdataset(
            temp_pet_fn,
            geom=self.region,
            buffer=1,
            time_tuple=(starttime, endtime),
            variables=variables,
            single_var_as_array=False,  # always return dataset
        )

        if chunksize is not None:
            ds = ds.chunk({"time": chunksize})

        dem_forcing = None
        if dem_forcing_fn != None:
            dem_forcing = self.data_catalog.get_rasterdataset(
                dem_forcing_fn,
                geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()

        temp_in = hydromt.workflows.forcing.temp(
            ds["temp"],
            dem_model=self.staticmaps[self._MAPS["bnd_1"]],
            dem_forcing=dem_forcing,
            lapse_correction=temp_correction,
            logger=self.logger,
            freq=None,  # resample time after pet workflow
            **kwargs,
        )

        if not skip_pet:
            pet_out = hydromt.workflows.forcing.pet(
                ds[variables[1:]],
                dem_model=self.staticmaps[self._MAPS["bnd_1"]],
                temp=temp_in,
                method=pet_method,
                press_correction=press_correction,
                freq=freq,
                resample_kwargs=dict(label="right", closed="right"),
                logger=self.logger,
                **kwargs,
            )
            # Update meta attributes with setup opt
            opt_attr = {
                "pet_fn": temp_pet_fn,
                "pet_method": pet_method,
            }
            pet_out.attrs.update(opt_attr)
            self.set_forcing(pet_out.where(mask), name="pet")

        # resample temp after pet workflow
        temp_out = hydromt.workflows.forcing.resample_time(
            temp_in,
            freq,
            upsampling="bfill",  # we assume right labeled original data
            downsampling="mean",
            label="right",
            closed="right",
            conserve_mass=False,
            logger=self.logger,
        )
        # Update meta attributes with setup opt (used for default naming later)
        opt_attr = {
            "temp_fn": temp_pet_fn,
            "temp_correction": str(temp_correction),
        }
        temp_out.attrs.update(opt_attr)
        self.set_forcing(temp_out.where(mask), name="temp")

    # I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_staticmaps()
        self.read_intbl()
        self.read_staticgeoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Write model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        self.write_data_catalog()
        if self.config:  # try to read default if not yet set
            self.write_config()
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._forcing:
            self.write_forcing()

    def read_staticmaps(self, **kwargs):
        """Read staticmaps"""
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read staticmaps from {fn}")
            # FIXME: we need a smarter (lazy) solution for big models which also
            # works when overwriting / appending data in the same source!
            ds = xr.open_dataset(fn, mask_and_scale=False, **kwargs).load()
            ds.close()
            self.set_staticmaps(ds)
        elif len(glob.glob(join(self.root, "staticmaps", "*.map"))) > 0:
            self.read_staticmaps_pcr()

    def write_staticmaps(self):
        """Write staticmaps"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        ds_out = self.staticmaps
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)
        # Check if all sub-folders in fn exists and if not create them
        if not isdir(dirname(fn)):
            os.makedirs(dirname(fn))
        self.logger.info(f"Write staticmaps to {fn}")
        mask = ds_out[self._MAPS["bnd_1"]] > 0
        for v in ds_out.data_vars:
            ds_out[v] = ds_out[v].where(mask, ds_out[v].raster.nodata)
        ds_out.to_netcdf(fn)
        # self.write_staticmaps_pcr()

    def read_staticmaps_pcr(self, crs=4326, **kwargs):
        """Read and staticmaps at <root/staticmaps> and parse to xarray"""
        if self._read and "chunks" not in kwargs:
            kwargs.update(chunks={"y": -1, "x": -1})
        fn = join(self.root, "staticmaps", f"*.map")
        fns = glob.glob(fn)
        if len(fns) == 0:
            self.logger.warning(f"No staticmaps found at {fn}")
            return
        self._staticmaps = open_mfraster(fns, **kwargs)
        for name in self.staticmaps.raster.vars:
            if PCR_VS_MAP.get(name, "scalar") == "bool":
                self._staticmaps[name] = self._staticmaps[name] == 1
                # a nodata value is required when writing
                self._staticmaps[name].raster.set_nodata(0)
        path = join(self.root, "staticmaps", "clim", f"LAI*")
        if len(glob.glob(path)) > 0:
            da_lai = open_mfraster(
                path, concat=True, concat_dim="time", logger=self.logger, **kwargs
            )
            self.set_staticmaps(da_lai, "LAI")

        # reorganize c_0 etc maps
        da_c = []
        list_c = [v for v in self._staticmaps if str(v).startswith("c_")]
        if len(list_c) > 0:
            for i, v in enumerate(list_c):
                da_c.append(self._staticmaps[f"c_{i:d}"])
            da = xr.concat(
                da_c, pd.Index(np.arange(len(list_c), dtype=int), name="layer")
            ).transpose("layer", ...)
            self.set_staticmaps(da, "c")

        if self.crs is None:
            if crs is None:
                crs = 4326  # default to 4326
            self.set_crs(crs)

    def write_staticmaps_pcr(self):
        """Write staticmaps at <root/staticmaps> in PCRaster maps format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        ds_out = self.staticmaps
        if "LAI" in ds_out.data_vars:
            ds_out = ds_out.rename_vars({"LAI": "clim/LAI"})
        if "c" in ds_out.data_vars:
            for layer in ds_out["layer"]:
                ds_out[f"c_{layer.item():d}"] = ds_out["c"].sel(layer=layer)
                ds_out[f"c_{layer.item():d}"].raster.set_nodata(
                    ds_out["c"].raster.nodata
                )
            ds_out = ds_out.drop_vars(["c", "layer"])
        self.logger.info("Writing (updated) staticmap files.")
        # add datatypes for maps with same basenames, e.g. gauges_grdc
        pcr_vs_map = PCR_VS_MAP.copy()
        for var_name in ds_out.raster.vars:
            base_name = "_".join(var_name.split("_")[:-1])  # clip _<postfix>
            if base_name in PCR_VS_MAP:
                pcr_vs_map.update({var_name: PCR_VS_MAP[base_name]})
        ds_out.raster.to_mapstack(
            root=join(self.root, "staticmaps"),
            mask=True,
            driver="PCRaster",
            pcr_vs_map=pcr_vs_map,
            logger=self.logger,
        )

    def read_staticgeoms(self):
        """Read and staticgeoms at <root/staticgeoms> and parse to geopandas"""
        if not self._write:
            self._staticgeoms = dict()  # fresh start in read-only mode
        dir_default = join(self.root, "staticmaps.nc")
        dir_mod = dirname(
            self.get_config("input.path_static", abs_path=True, fallback=dir_default)
        )
        fns = glob.glob(join(dir_mod, "staticgeoms", "*.geojson"))
        if len(fns) > 1:
            self.logger.info("Reading model staticgeom files.")
        for fn in fns:
            name = basename(fn).split(".")[0]
            if name != "region":
                self.set_staticgeoms(gpd.read_file(fn), name=name)

    def write_staticgeoms(self):
        """Write staticmaps at <root/staticgeoms> in model ready format"""
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.staticgeoms:
            self.logger.info("Writing model staticgeom to file.")
            for name, gdf in self.staticgeoms.items():
                fn_out = join(self.root, "staticgeoms", f"{name}.geojson")
                gdf.to_file(fn_out, driver="GeoJSON")

    def read_forcing(self):
        """Read forcing"""
        fn_default = join(self.root, "inmaps.nc")
        fn = self.get_config("input.path_forcing", abs_path=True, fallback=fn_default)
        if not self._write:
            # start fresh in read-only mode
            self._forcing = dict()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read forcing from {fn}")
            ds = xr.open_dataset(fn, chunks={"time": 30})
            for v in ds.data_vars:
                self.set_forcing(ds[v])

    def write_forcing(
        self,
        fn_out=None,
        freq_out=None,
        chunksize=1,
        decimals=2,
        time_units="days since 1900-01-01T00:00:00",
        **kwargs,
    ):
        """write forcing at `fn_out` in model ready format.

        If no `fn_out` path is provided and path_forcing from the  wflow toml exists,
        the following default filenames are used:

            * Default name format (with downscaling): inmaps_sourcePd_sourceTd_methodPET_freq_startyear_endyear.nc
            * Default name format (no downscaling): inmaps_sourceP_sourceT_methodPET_freq_startyear_endyear.nc

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output netcdf file; if None the name is read from the wflow
            toml file.
        freq_out: str (Offset), optional
            Write several files for the forcing according to fn_freq. For example 'Y' for one file per year or 'M'
            for one file per month. By default writes the one file.
            For more options, see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        chunksize: int, optional
            Chunksize on time dimension when saving to disk. By default 1.
        decimals, int, optional
            Round the ouput data to the given number of decimals.
        time_units: str, optional
            Common time units when writting several netcdf forcing files. By default "days since 1900-01-01T00:00:00".

        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.forcing:
            self.logger.info("Write forcing file")

            # Get default forcing name from forcing attrs
            yr0 = pd.to_datetime(self.get_config("starttime")).year
            yr1 = pd.to_datetime(self.get_config("endtime")).year
            freq = self.get_config("timestepsecs")

            # get output filename
            if fn_out is not None:
                self.set_config("input.path_forcing", fn_out)
                self.write_config()  # re-write config
            else:
                fn_out = self.get_config("input.path_forcing", abs_path=True)
                # get deafult filename if file exists
                if fn_out is None or isfile(fn_out):
                    self.logger.warning(
                        "Netcdf forcing file from input.path_forcing in the TOML  "
                        "already exists, using default name."
                    )
                    sourceP = ""
                    sourceT = ""
                    methodPET = ""
                    if "precip" in self.forcing:
                        val = self.forcing["precip"].attrs.get("precip_clim_fn", None)
                        Pdown = "d" if val is not None else ""
                        val = self.forcing["precip"].attrs.get("precip_fn", None)
                        if val is not None:
                            sourceP = f"_{val}{Pdown}"
                    if "temp" in self.forcing:
                        val = self.forcing["temp"].attrs.get("temp_correction", "False")
                        Tdown = "d" if val == "True" else ""
                        val = self.forcing["temp"].attrs.get("temp_fn", None)
                        if val is not None:
                            sourceT = f"_{val}{Tdown}"
                    if "pet" in self.forcing:
                        val = self.forcing["pet"].attrs.get("pet_method", None)
                        if val is not None:
                            methodPET = f"_{val}"
                    fn_default = (
                        f"inmaps{sourceP}{sourceT}{methodPET}_{freq}_{yr0}_{yr1}.nc"
                    )
                    fn_default_path = join(self.root, fn_default)
                    if isfile(fn_default_path):
                        self.logger.warning(
                            "Netcdf default forcing file already exists, skipping write_forcing. "
                            "To overwrite netcdf forcing file: change name input.path_forcing "
                            "in setup_config section of the build inifile."
                        )
                        return
                    else:
                        self.set_config("input.path_forcing", fn_default)
                        self.write_config()  # re-write config
                        fn_out = fn_default_path

            # merge, process and write forcing
            ds = xr.merge(self.forcing.values())
            if decimals is not None:
                ds = ds.round(decimals)
            # write with output chunksizes with single timestep and complete
            # spatial grid to speed up the reading from wflow.jl
            # dims are always ordered (time, y, x)
            ds.raster._check_dimensions()
            chunksizes = (chunksize, ds.raster.ycoords.size, ds.raster.xcoords.size)
            encoding = {
                v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
                for v in ds.data_vars.keys()
            }
            # make sure no _FillValue is written to the time dimension
            # For several forcing files add common units attributes to time
            ds["time"].attrs.pop("_FillValue", None)
            encoding["time"] = {"_FillValue": None}

            # Check if all sub-folders in fn_out exists and if not create them
            if not isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))

            forcing_list = []

            if freq_out is None:
                # with compute=False we get a delayed object which is executed when
                # calling .compute where we can pass more arguments to the dask.compute method
                forcing_list.append([fn_out, ds])
            else:
                self.logger.info(f"Writting several forcing with freq {freq_out}")
                # For several forcing files add common units attributes to time
                encoding["time"] = {"_FillValue": None, "units": time_units}
                # Updating path forcing in config
                fns_out = os.path.relpath(fn_out, self.root)
                fns_out = f"{str(fns_out)[0:-3]}_*.nc"
                self.set_config("input.path_forcing", fns_out)
                for label, ds_gr in ds.resample(time=freq_out):
                    # ds_gr = group[1]
                    start = ds_gr["time"].dt.strftime("%Y%m%d")[0].item()
                    fn_out_gr = f"{str(fn_out)[0:-3]}_{start}.nc"
                    forcing_list.append([fn_out_gr, ds_gr])

            for fn_out_gr, ds_gr in forcing_list:
                self.logger.info(f"Process forcing; saving to {fn_out_gr}")
                delayed_obj = ds_gr.to_netcdf(
                    fn_out_gr, encoding=encoding, mode="w", compute=False
                )
                with ProgressBar():
                    delayed_obj.compute(**kwargs)

            # TO profile uncomment lines below to replace lines above
            # from dask.diagnostics import Profiler, CacheProfiler, ResourceProfiler
            # import cachey
            # with Profiler() as prof, CacheProfiler(metric=cachey.nbytes) as cprof, ResourceProfiler() as rprof:
            #     delayed_obj.compute()
            # visualize([prof, cprof, rprof], file_path=r'c:\Users\eilan_dk\work\profile2.html')

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        if not self._write:
            # start fresh in read-only mode
            self._states = dict()
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray/xr.Dataset"""
        if not self._write:
            # start fresh in read-only mode
            self._results = dict()

        # Read gridded netcdf (output section)
        nc_fn = self.get_config("output.path", abs_path=True)
        if nc_fn is not None and isfile(nc_fn):
            self.logger.info(f"Read results from {nc_fn}")
            ds = xr.open_dataset(nc_fn, chunks={"time": 30})
            # TODO ? align coords names and values of results nc with staticmaps
            self.set_results(ds, name="output")

        # Read scalar netcdf (netcdf section)
        ncs_fn = self.get_config("netcdf.path", abs_path=True)
        if ncs_fn is not None and isfile(ncs_fn):
            self.logger.info(f"Read results from {ncs_fn}")
            ds = xr.open_dataset(ncs_fn, chunks={"time": 30})
            self.set_results(ds, name="netcdf")

        # Read csv timeseries (csv section)
        csv_fn = self.get_config("csv.path", abs_path=True)
        if csv_fn is not None and isfile(csv_fn):
            csv_dict = utils.read_csv_results(
                csv_fn, config=self.config, maps=self.staticmaps
            )
            for key in csv_dict:
                # Add to results
                self.set_results(csv_dict[f"{key}"])

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_intbl(self, **kwargs):
        """Read and intbl files at <root/intbl> and parse to xarray"""
        if not self._write:
            self._intbl = dict()  # start fresh in read-only mode
        if not self._read:
            self.logger.info("Reading default intbl files.")
            fns = glob.glob(join(DATADIR, "imod", "intbl", f"*.tbl"))
        else:
            self.logger.info("Reading model intbl files.")
            fns = glob.glob(join(self.root, "intbl", f"*.tbl"))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn, delim_whitespace=True, header=None)
                tbl.columns = [
                    f"expr{i+1}" if i + 1 < len(tbl.columns) else "value"
                    for i in range(len(tbl.columns))
                ]  # rename columns
                self.set_intbl(tbl, name=name)

    def write_intbl(self):
        """Write intbl at <root/intbl> in PCRaster table format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.intbl:
            self.logger.info("Writing intbl files.")
            for name in self.intbl:
                fn_out = join(self.root, "intbl", f"{name}.tbl")
                self.intbl[name].to_csv(fn_out, sep=" ", index=False, header=False)

    def set_intbl(self, df, name):
        """Add intbl <pandas.DataFrame> to model."""
        if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
            raise ValueError("df type not recognized, should be pandas.DataFrame.")
        if name in self._intbl:
            if not self._write:
                raise IOError(f"Cannot overwrite intbl {name} in read-only mode")
            elif self._read:
                self.logger.warning(f"Overwriting intbl: {name}")
        self._intbl[name] = df

    def _configread(self, fn):
        with codecs.open(fn, "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        return fdict

    def _configwrite(self, fn):
        with codecs.open(fn, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)

    ## WFLOW specific data and methods

    @property
    def intbl(self):
        """Returns a dictionary of pandas.DataFrames representing the wflow intbl files."""
        if not self._intbl:
            self.read_intbl()
        return self._intbl

    @property
    def flwdir(self):
        """Returns the pyflwdir.FlwdirRaster object parsed from the wflow ldd."""
        if self._flwdir is None:
            self.set_flwdir()
        return self._flwdir

    def set_flwdir(self, ftype="infer"):
        """Parse pyflwdir.FlwdirRaster object parsed from the wflow ldd"""
        flwdir_name = flwdir_name = self._MAPS["flwdir"]
        self._flwdir = flw.flwdir_from_da(
            self.staticmaps[flwdir_name],
            ftype=ftype,
            check_ftype=True,
            mask=(self.staticmaps[self._MAPS["bnd_1"]] > 0),
        )

    @property
    def basins(self):
        """Returns a basin(s) geometry as a geopandas.GeoDataFrame."""
        if "basins" in self.staticgeoms:
            gdf = self.staticgeoms["basins"]
        else:
            gdf = flw.basin_shape(
                self.staticmaps, self.flwdir, basin_name=self._MAPS["bnd_1"]
            )
            self.set_staticgeoms(gdf, name="basins")
        return gdf

    @property
    def rivers(self):
        """Returns a river geometry as a geopandas.GeoDataFrame. If available, the
        stream order and upstream area values are added to the geometry properties.
        """
        if "rivers" in self.staticgeoms:
            gdf = self.staticgeoms["rivers"]
        elif self._MAPS["rivmsk"] in self.staticmaps:
            rivmsk = self.staticmaps[self._MAPS["rivmsk"]].values != 0
            # Check if there are river cells in the model before continuing
            if np.any(rivmsk):
                # add stream order 'strord' column
                strord = self.flwdir.stream_order(mask=rivmsk)
                feats = self.flwdir.streams(mask=rivmsk, strord=strord)
                gdf = gpd.GeoDataFrame.from_features(feats)
                gdf.crs = pyproj.CRS.from_user_input(self.crs)
                self.set_staticgeoms(gdf, name="rivers")
            else:
                self.logger.warning("No river cells detected in the selected basin.")
                gdf = None
        return gdf

    def clip_staticmaps(
        self,
        region,
        buffer=0,
        align=None,
        crs=4326,
    ):
        """Clip staticmaps to subbasin.

        Parameters
        ----------
        region : dict
            See :meth:`models.wflow.WflowModel.setup_basemaps`
        buffer : int, optional
            Buffer around subbasin in number of pixels, by default 0
        align : float, optional
            Align bounds of region to raster with resolution <align>, by default None
        crs: int, optional
            Default crs of the staticmaps to clip.

        Returns
        -------
        xarray.DataSet
            Clipped staticmaps.
        """
        basins_name = self._MAPS["bnd_1"]
        flwdir_name = self._MAPS["flwdir"]

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        # translate basin and outlet kinds to geom
        geom = region.get("geom", None)
        bbox = region.get("bbox", None)
        if kind in ["basin", "outlet", "subbasin"]:
            # supply bbox to avoid getting basin bounds first when clipping subbasins
            if kind == "subbasin" and bbox is None:
                region.update(bbox=self.bounds)
            geom, _ = hydromt.workflows.get_basin_geometry(
                ds=self.staticmaps,
                logger=self.logger,
                kind=kind,
                basins_name=basins_name,
                flwdir_name=flwdir_name,
                **region,
            )

        # clip based on subbasin args, geom or bbox
        if geom is not None:
            ds_staticmaps = self.staticmaps.raster.clip_geom(
                geom, align=align, buffer=buffer
            )
            ds_staticmaps[basins_name] = ds_staticmaps[basins_name].where(
                ds_staticmaps["mask"], self.staticmaps[basins_name].raster.nodata
            )
            ds_staticmaps[basins_name].attrs.update(
                _FillValue=self.staticmaps[basins_name].raster.nodata
            )
        elif bbox is not None:
            ds_staticmaps = self.staticmaps.raster.clip_bbox(
                bbox, align=align, buffer=buffer
            )

        # Update flwdir staticmaps and staticgeoms
        if self.crs is None and crs is not None:
            self.set_crs(crs)

        self._staticmaps = xr.Dataset()
        self.set_staticmaps(ds_staticmaps)

        # add pits at edges after clipping
        self._flwdir = None  # make sure old flwdir object is removed
        self.staticmaps[self._MAPS["flwdir"]].data = self.flwdir.to_array("ldd")

        self._staticgeoms = dict()
        self.basins
        self.rivers

        # Update reservoir and lakes
        remove_reservoir = False
        if self._MAPS["resareas"] in self.staticmaps:
            reservoir = self.staticmaps[self._MAPS["resareas"]]
            if not np.any(reservoir > 0):
                remove_reservoir = True
                remove_maps = [
                    self._MAPS["resareas"],
                    self._MAPS["reslocs"],
                    "ResSimpleArea",
                    "ResDemand",
                    "ResTargetFullFrac",
                    "ResTargetMinFrac",
                    "ResMaxRelease",
                    "ResMaxVolume",
                ]
                self._staticmaps = self.staticmaps.drop_vars(remove_maps)

        remove_lake = False
        if self._MAPS["lakeareas"] in self.staticmaps:
            lake = self.staticmaps[self._MAPS["lakeareas"]]
            if not np.any(lake > 0):
                remove_lake = True
                remove_maps = [
                    self._MAPS["lakeareas"],
                    self._MAPS["lakelocs"],
                    "LinkedLakeLocs",
                    "LakeStorFunc",
                    "LakeOutflowFunc",
                    "LakeArea",
                    "LakeAvgLevel",
                    "LakeAvgOut",
                    "LakeThreshold",
                    "Lake_b",
                    "Lake_e",
                ]
                self._staticmaps = self.staticmaps.drop_vars(remove_maps)

        # Update config
        # Remove the absolute path and if needed remove lakes and reservoirs
        if remove_reservoir:
            # change reservoirs = true to false
            self.set_config("model.reservoirs", False)
            # remove states
            if self.get_config("state.lateral.river.reservoir") is not None:
                del self.config["state"]["lateral"]["river"]["reservoir"]

        if remove_lake:
            # change lakes = true to false
            self.set_config("model.lakes", False)
            # remove states
            if self.get_config("state.lateral.river.lake") is not None:
                del self.config["state"]["lateral"]["river"]["lake"]

    def clip_forcing(self, crs=4326, **kwargs):
        """Return clippped forcing for subbasin.

        Returns
        -------
        xarray.DataSet
            Clipped forcing.

        """
        if len(self.forcing) > 0:
            self.logger.info("Clipping NetCDF forcing..")
            ds_forcing = xr.merge(self.forcing.values()).raster.clip_bbox(
                self.staticmaps.raster.bounds
            )
            self.set_forcing(ds_forcing)
