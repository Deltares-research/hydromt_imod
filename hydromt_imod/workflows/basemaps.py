# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import logging
import geopandas as gpd
import pandas as pd
from scipy.optimize import curve_fit
import pyflwdir
from hydromt import flw, gis_utils
from typing import Optional

from . import ptf

logger = logging.getLogger(__name__)

__all__ = ["hydrography", "topography", "soilgrids"]

def hydrography(
    geology_fn,
    ds: xr.Dataset,
    res: float,
    xy: Optional[gpd.GeoDataFrame] = None,    
    upscale_method: str = "ihu",
    flwdir_name: str = "flwdir",
    uparea_name: str = "uparea",
    basins_name: str = "basins",
    strord_name: str = "strord",
    ftype: str = "infer",
    logger=logger,
):
    """Returns hydrography maps (see list below) and FlwdirRaster object based on 
    gridded flow direction and elevation data input. 

    The output maps are:\
    - flwdir : flow direction [-]\
    - basins : basin map [-]\
    - uparea : upstream area [km2]\
    - strord : stream order [-]\

    If the resolution is lower than the source resolution, the flow direction data is 
    upscaled and river length and slope are based on subgrid flow paths and the following
    maps are added:\
    - subare : contributing area to each subgrid outlet pixel (unit catchment area) [km2]\
    - subelv : elevation at subgrid outlet pixel [m+REF]\

    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    res : float
        output resolution
    xy : geopandas.GeoDataFrame, optional
        Subbasin pits. Only required when upscaling a subbasin.
    upscale_method : {'ihu', 'eam', 'dmm'}
        Upscaling method for flow direction data, by default 'ihu', see [1]_
    ftype : {'d8', 'ldd', 'nextxy', 'nextidx', 'infer'}, optional
        name of flow direction type, infer from data if 'infer', by default is 'infer'
    flwdir_name, elevtn_name, uparea_name, basins_name, strord_name : str, optional
        Name of flow direction [-], elevation [m], upstream area [km2], basin index [-] 
        and stream order [-] variables in ds

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.

    References
    ----------
    .. [1] Eilander et al. (2021). A hydrography upscaling method for scale-invariant parametrization of distributed hydrological models. 
           Hydrology and Earth System Sciences, 25(9), 5287–5313. https://doi.org/10.5194/hess-25-5287-2021

    See Also
    --------
    pyflwdir.FlwdirRaster.upscale_flwdir
    """
    # TODO add check if flwdir in ds, calculate if not
    flwdir = None
    basins = None
    outidx = None
    if not "mask" in ds.coords and xy is None:
        ds.coords["mask"] = xr.Variable(
            dims=ds.raster.dims, data=np.ones(ds.raster.shape, dtype=np.bool)
        )
    elif not "mask" in ds.coords:
        # NOTE if no subbasin mask is provided calculate it here
        logger.debug(f"Delineate {xy[0].size} subbasin(s).")
        flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype)
        basins = flwdir.basins(xy=xy).astype(np.int32)
        ds.coords["mask"].data = basins != 0
        if not np.any(ds.coords["mask"]):
            raise ValueError("Delineating subbasins not successfull.")
    elif xy is not None:
        # NOTE: this mask is passed on from get_basin_geometry method
        logger.debug(f"Mask in dataset assumed to represent subbasins.")
    ncells = np.sum(ds["mask"].values)
    logger.debug(f"(Sub)basin at original resolution has {ncells} cells.")

    scale_ratio = int(np.round(res / ds.raster.res[0]))
    if scale_ratio > 1:  # upscale flwdir
        if flwdir is None:
            # NOTE initialize with mask is FALSE
            flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=False)
        if xy is not None:
            logger.debug(f"Burn subbasin outlet in upstream area data.")
            if isinstance(xy, gpd.GeoDataFrame):
                assert xy.crs == ds.raster.crs
                xy = xy.geometry.x, xy.geometry.y
            idxs_pit = flwdir.index(*xy)
            flwdir.add_pits(idxs=idxs_pit)
            uparea = ds[uparea_name].values
            uparea.flat[idxs_pit] = uparea.max() + 1.0
            ds[uparea_name].data = uparea
        logger.info(
            f"Upscale flow direction data: {scale_ratio:d}x, {upscale_method} method."
        )
        da_flw, flwdir_out = flw.upscale_flwdir(
            ds,
            flwdir=flwdir,
            scale_ratio=scale_ratio,
            method=upscale_method,
            uparea_name=uparea_name,
            flwdir_name=flwdir_name,
            logger=logger,
        )
        da_flw.raster.set_crs(ds.raster.crs)
        # make sure x_out and y_out get saved
        ds_out = da_flw.to_dataset().reset_coords(["x_out", "y_out"])
        dims = ds_out.raster.dims
        # find pits within basin mask
        idxs_pit0 = flwdir_out.idxs_pit
        outlon = ds_out["x_out"].values.ravel()
        outlat = ds_out["y_out"].values.ravel()
        sel = {
            ds.raster.x_dim: xr.Variable("yx", outlon[idxs_pit0]),
            ds.raster.y_dim: xr.Variable("yx", outlat[idxs_pit0]),
        }
        outbas_pit = ds.coords["mask"].sel(sel, method="nearest").values
        # derive basins
        if np.any(outbas_pit != 0):
            idxs_pit = idxs_pit0[outbas_pit != 0]
            basins = flwdir_out.basins(idxs=idxs_pit).astype(np.int32)
            ds_out.coords["mask"] = xr.Variable(
                dims=ds_out.raster.dims, data=basins != 0, attrs=dict(_FillValue=0)
            )
        else:
            # This is a patch for basins which are clipped based on bbox or wrong geom
            ds_out.coords["mask"] = (
                ds["mask"]
                .astype(np.int8)
                .raster.reproject_like(da_flw, method="nearest")
                .astype(np.bool)
            )
            basins = ds_out["mask"].values.astype(np.int32)
            logger.warning(
                "The basin delination might be wrong as no original resolution outlets "
                "are found in the upscaled map."
            )
        ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        # calculate upstream area using subgrid ucat cell areas
        outidx = np.where(
            ds_out["mask"], da_flw.coords["idx_out"].values, flwdir_out._mv
        )
        subare = flwdir.ucat_area(outidx, unit="km2")[1]
        uparea = flwdir_out.accuflux(subare)
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs)
        # NOTE: subgrid cella area is currently not used in wflow
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs)
        if "elevtn" in ds:
            subelv = ds["elevtn"].values.flat[outidx]
            subelv = np.where(outidx >= 0, subelv, -9999)
            attrs = dict(_FillValue=-9999, unit="m+REF")
            ds_out["subelv"] = xr.Variable(dims, subelv, attrs=attrs)
        # initiate masked flow dir
        flwdir_out = flw.flwdir_from_da(
            ds_out[flwdir_name], ftype=flwdir.ftype, mask=True
        )
                
    else:
        # NO upscaling : source resolution equals target resolution
        # NOTE (re-)initialize with mask is TRUE
        ftype = flwdir.ftype if flwdir is not None and ftype == "infer" else ftype
        flwdir = flw.flwdir_from_da(ds[flwdir_name], ftype=ftype, mask=True)
        flwdir_out = flwdir
        ds_out = xr.DataArray(
            name=flwdir_name,
            data=flwdir_out.to_array(),
            coords=ds.raster.coords,
            dims=ds.raster.dims,
            attrs=dict(
                long_name=f"{ftype} flow direction",
                _FillValue=flwdir_out._core._mv,
            ),
        ).to_dataset()
        dims = ds_out.raster.dims
        ds_out.coords["mask"] = xr.Variable(
            dims=dims, data=flwdir_out.mask.reshape(flwdir_out.shape)
        )
        # copy data variables from source if available
        for dvar in [basins_name, uparea_name, strord_name]:
            if dvar in ds.data_vars:
                ds_out[dvar] = xr.where(
                    ds_out["mask"],
                    ds[dvar],
                    ds[dvar].dtype.type(ds[dvar].raster.nodata),
                )
                ds_out[dvar].attrs.update(ds[dvar].attrs)
        # basins
        if basins_name not in ds_out.data_vars:
            if basins is None:
                basins = flwdir_out.basins(idxs=flwdir_out.idxs_pit).astype(np.int32)
            ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        # upstream area
        if uparea_name not in ds_out.data_vars:
            uparea = flwdir_out.upstream_area("km2")  # km2
            attrs = dict(_FillValue=-9999, unit="km2")
            ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs)
        # cell area
        # NOTE: subgrid cella area is currently not used in wflow
        ys, xs = ds.raster.ycoords.values, ds.raster.xcoords.values
        subare = gis_utils.reggrid_area(ys, xs) / 1e6  # km2
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs)
        
    if geology_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm
      
    for i in range(0, len(soildepth_cm)):
        ds_out["bnd_" + str(i+1)] = ds_out["mask"]
        
    # logging
    npits = flwdir_out.idxs_pit.size
    xy_pit = flwdir_out.xy(flwdir_out.idxs_pit[:5])
    xy_pit_str = ", ".join([f"({x:.5f},{y:.5f})" for x, y in zip(*xy_pit)])
    # stream order
    if strord_name not in ds_out.data_vars:
        logger.debug(f"Derive stream order.")
        strord = flwdir_out.stream_order()
        ds_out[strord_name] = xr.Variable(dims, strord, attrs=dict(_FillValue=-1))

    # clip to basin extent
    ds_out = ds_out.raster.clip_mask(mask=ds_out[basins_name])
    ds_out.raster.set_crs(ds.raster.crs)
    logger.debug(
        f"Map shape: {ds_out.raster.shape}; active cells: {flwdir_out.ncells}."
    )
    logger.debug(f"Outlet coordinates ({len(xy_pit[0])}/{npits}): {xy_pit_str}.")
    if np.any(np.asarray(ds_out.raster.shape) == 1):
        raise ValueError(
            "The output extent should at consist of two cells on each axis. "
            "Consider using a larger domain or higher spatial resolution. "
            "For subbasin models, consider a (higher) threshold to snap the outlet."
        )
        
    ds_out = ds_out.drop_vars(['x_out','y_out', 'idx_out', 'mask', 'uparea', 'subare', 'subelv', 'basins', 'flwdir'])
    return ds_out#, flwdir_out


def topography(
    geology_fn,
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    elevtn_name: str = "elevtn",
    method: str = "average",
    logger=logger,
):
    """Returns topography maps (see list below) at model resolution based on gridded 
    elevation data input. 

    The following topography maps are calculated:\
    - elevtn : average elevation [m]\
    - lndslp : average land surface slope [m/m]\
    
    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    elevtn_name, bot_name : str, optional
        Name of elevation [m] and land surface slope [m/m] variables in ds
    method: str, optional
        Resample method used to reproject the input data, by default "average"

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.

    See Also
    --------
    pyflwdir.dem.slope
    """
    # clip or reproject if non-identical grids
    
    if geology_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm
    
    for i in range(0, len(soildepth_cm)):
        ds["top_" + str(i+1)] = ds["elevtn"]-soildepth_cm[i]
        
    for i in range(0, len(soildepth_cm)-1):    
        ds["bot_" + str(i+1)] = ds["elevtn"]-soildepth_cm[i+1]
        
    ds["bot_" + str(len(soildepth_cm))] = ds["bot_" + str(len(soildepth_cm)-1)]-200 
    
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    
    for i in range(0, len(soildepth_cm)):
        ds_out["top_" + str(i+1)] = ds["top_" + str(i+1)].raster.reproject_like(ds_like, method)
        ds_out["bot_" + str(i+1)] = ds["bot_" + str(i+1)].raster.reproject_like(ds_like, method)
    
    #ds_out = ds.raster.reproject_like(ds_like, method)
    #ds_out = ds[[elevtn_name]]
    #ds_out = ds_out.drop_vars(['elevtn'])
    return ds_out
    
def average_soillayers_block(geology_fn, ds, soilthickness):
    """
    Determine the weighted average of soil property at different depths over soil thickness,
    assuming that the properties are computed at the mid-point of the interval and are considered
    constant over the whole depth interval (Sousa et al., 2020). https://doi.org/10.5194/soil-2020-65
    This function is used for soilgrids_2020.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property over each soil depth profile [sl1 - sl6].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing weighted average of soil property.

    """
    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)
    
    if geology_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm

    for i in ds.sl:

        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x / y,
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av


def average_soillayers(geology_fn, ds, soilthickness):

    """
    Determine weighted average of soil property at different depths over soil thickness,
    using the trapezoidal rule.
    See also: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez, M., Kilibarda,
    M., Blagotic, A., et al.: SoilGrids250m: Global gridded soil information based on machine learning,
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    This function is used for soilgrids (2017).

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property at each soil depth [sl1 - sl7].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing weighted average of soil property.

    """

    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)
    
    if geology_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm

    for i in range(1, len(ds.sl)):  # range(1, 7):

        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x * (1 / (y * 2)),
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av


def pore_size_distrution_index_layers(ds, thetas):

    """
    Determine pore size distribution index per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing pore size distribution index [-] for each soil layer
        depth.

    """

    ds_out = xr.apply_ufunc(
        ptf.pore_size_index_brakensiek,
        ds["sndppt"],
        thetas,
        ds["clyppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    ds_out.name = "pore_size"
    ds_out.raster.set_nodata(np.nan)
    # ds_out = ds_out.raster.interpolate_na("nearest")
    return ds_out


def kv_layers(ds, thetas, ptf_name):

    """
    Determine vertical saturated hydraulic conductivity (KsatVer) per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.
    ptf_name : str
        PTF to use for calculation KsatVer.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing KsatVer [mm/day] for each soil layer depth.
    """
    if ptf_name == "brakensiek":
        ds_out = xr.apply_ufunc(
            ptf.kv_brakensiek,
            thetas,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif ptf_name == "cosby":
        ds_out = xr.apply_ufunc(
            ptf.kv_cosby,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )

    ds_out.name = "kv"
    ds_out.raster.set_nodata(np.nan)

    return ds_out


def func(x, b):
    return np.exp(-b * x)


def do_linalg(x, y):
    """
    Apply np.linalg.lstsq and return fitted parameter.

    Parameters
    ----------
    x : array_like (float)
        “Coefficient” matrix.
    y : array_like (float)
        dependent variable.

    Returns
    -------
    popt_0 : float
        Optimal value for the parameter fit.

    """
    idx = ((~np.isinf(np.log(y)))) & ((~np.isnan(y)))
    return np.linalg.lstsq(x[idx, np.newaxis], np.log(y[idx]), rcond=None)[0][0]


def do_curve_fit(x, y):
    """
    Apply scipy.optimize.curve_fit and return fitted parameter. If least-squares minimization
    fails with an inital guess p0 of 1e-3, and 1e-4, np.linalg.lstsq is used for curve fitting.

    Parameters
    ----------
    x : array_like of M length (float)
        independent variable.
    y : array_like of M length (float)
        dependent variable.

    Returns
    -------
    popt_0 : float
        Optimal value for the parameter fit.

    """
    idx = ((~np.isinf(np.log(y)))) & ((~np.isnan(y)))
    if len(y[idx]) == 0:
        popt_0 = np.nan
    else:
        try:
            # try curve fitting with certain p0
            popt_0 = curve_fit(func, x[idx], y[idx], p0=(1e-3))[0]
        except RuntimeError:
            try:
                # try curve fitting with lower p0
                popt_0 = curve_fit(func, x[idx], y[idx], p0=(1e-4))[0]
            except RuntimeError:
                # do linalg  regression instead
                popt_0 = np.linalg.lstsq(
                    x[idx, np.newaxis], np.log(y[idx]), rcond=None
                )[0][0]
    return popt_0


def soilgrids(ds, ds_like, ptfKsatVer, geology_fn, logger=logger):

    """
    Returns soil parameter maps at model resolution based on soil properties from SoilGrids datasets.
    Both soilgrids 2017 and 2020 are supported. Soilgrids 2017 provides soil properties at 7 specific depths, while soilgrids_2020 provides soil properties averaged over 6 depth intervals. 
    Ref: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez, M., Kilibarda, 
    M., Blagotic, A., et al.: SoilGrids250m: Global gridded soil information based on machine learning, 
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    Ref: de Sousa, L.M., Poggio, L., Batjes, N.H., Heuvelink, G., Kempen, B., Riberio, E. and Rossiter, D., 2020. 
    SoilGrids 2.0: producing quality-assessed soil information for the globe. SOIL Discussions, pp.1-37.
    https://doi.org/10.5194/soil-2020-65 

    The following soil parameter maps are calculated:\
    - thetaS            : average saturated soil water content [m3/m3]\
    - thetaR            : average residual water content [m3/m3]\
    - KsatVer           : vertical saturated hydraulic conductivity at soil surface [mm/day]\
    - SoilThickness     : soil thickness [mm]\
    - SoilMinThickness  : minimum soil thickness [mm] (equal to SoilThickness)\
    - khh_[z]cm         : KsatVer [mm/day] at soil depths [z] of SoilGrids data [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]\
    - soil              : USDA Soil texture based on percentage clay, silt, sand mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, 6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam]\
                
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    ptfKsatVer : str
        PTF to use for calculcation KsatVer.
    geology_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded soil parameters.
    """

    if geology_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm

    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # set nodata values in dataset to NaN (based on soil property SLTPPT at first soil layer)
    # ds = xr.where(ds["sltppt_sl1"] == ds["sltppt_sl1"].raster.nodata, np.nan, ds)
    ds = ds.raster.mask_nodata()

    # add new coordinate sl to merge layers
    ds = ds.assign_coords(sl=np.arange(1, len(soildepth_cm_midpoint) + 1))

    for var in ["bd", "oc", "ph", "clyppt", "sltppt", "sndppt"]:
        da_prop = []
        for i in np.arange(1, len(soildepth_cm_midpoint) + 1):
            da_prop.append(ds[f"{var}_sl{i}"])
            # remove layer from ds
            ds = ds.drop_vars(f"{var}_sl{i}")
        da = xr.concat(
            da_prop,
            pd.Index(
                np.arange(1, len(soildepth_cm_midpoint) + 1, dtype=int), name="sl"
            ),
        ).transpose("sl", ...)
        da.name = var
        # add concat maps to ds
        ds[f"{var}"] = da

    logger.info("calculate and resample thetaS")
    thetas_sl = xr.apply_ufunc(
        ptf.thetas_toth,
        ds["ph"],
        ds["bd"],
        ds["clyppt"],
        ds["sltppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    #if geology_fn == "soilgrids_2020":
    #    thetas = average_soillayers_block(thetas_sl, ds["soilthickness"])
    #else:
    #    thetas = average_soillayers(thetas_sl, ds["soilthickness"])
    #thetas = thetas.raster.reproject_like(ds_like, method="average")
    #ds_out["thetaS"] = thetas.astype(np.float32)

    logger.info("calculate and resample thetaR")
    thetar_sl = xr.apply_ufunc(
        ptf.thetar_toth,
        ds["oc"],
        ds["clyppt"],
        ds["sltppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    #if geology_fn == "soilgrids_2020":
    #    thetar = average_soillayers_block(thetar_sl, ds["soilthickness"])
    #else:
    #    thetar = average_soillayers(thetar_sl, ds["soilthickness"])
    #thetar = thetar.raster.reproject_like(ds_like, method="average")
    #ds_out["thetaR"] = thetar.astype(np.float32)

    soilthickness_hr = ds["soilthickness"]
    soilthickness = soilthickness_hr.raster.reproject_like(ds_like, method="average")
    # avoid zero soil thickness
    soilthickness = soilthickness.where(soilthickness > 0.0, np.nan)
    soilthickness.raster.set_nodata(np.nan)
    soilthickness = soilthickness.astype(np.float32)
    ds_out["SoilThickness"] = soilthickness * 10.0  # from [cm] to [mm]
    ds_out["SoilMinThickness"] = xr.DataArray.copy(ds_out["SoilThickness"], deep=False)

    logger.info("calculate and resample KsatVer")
    kv_sl_hr = kv_layers(ds, thetas_sl, ptfKsatVer)
    kv_sl = np.log(kv_sl_hr)
    kv_sl = kv_sl.raster.reproject_like(ds_like, method="average")
    kv_sl = np.exp(kv_sl)
    
    ds_out["KsatVer"] = kv_sl.sel(sl=1).astype(np.float32)

    for i, sl in enumerate(kv_sl["sl"]):
        kv = kv_sl.sel(sl=sl)
        ds_out["khh_" + str(i+1)] = kv.astype(
            np.float32
        )
        ds_out["kvv_" + str(i+1)] = kv.astype(
            np.float32
        )/2

    kv = kv_sl / kv_sl.sel(sl=1)
    logger.info("fit z - log(KsatVer) with numpy linalg regression (y = b*x) -> M_")
    popt_0_ = xr.apply_ufunc(
        do_linalg,
        soildepth_mm_midpoint_surface,
        kv.compute(),
        vectorize=True,
        dask="parallelized",
        input_core_dims=[["z"], ["sl"]],
        output_dtypes=[float],
        keep_attrs=True,
    )
  
    # wflow soil map is based on USDA soil classification
    # soilmap = ds["tax_usda"].raster.interpolate_na()
    # soilmap = soilmap.raster.reproject_like(ds_like, method="mode")
    # ds_out["soil"] = soilmap.astype(np.float32)

    # soil map is based on soil texture calculated with percentage sand, silt, clay
    # clay, silt percentages averaged over soil thickness
    #if geology_fn == "soilgrids_2020":
    #    clay_av = average_soillayers_block(ds["clyppt"], ds["soilthickness"])
    #    silt_av = average_soillayers_block(ds["sltppt"], ds["soilthickness"])
    #else:
    #    clay_av = average_soillayers(ds["clyppt"], ds["soilthickness"])
    #    silt_av = average_soillayers(ds["sltppt"], ds["soilthickness"])

    # calc soil texture
    #soil_texture = xr.apply_ufunc(
    #    ptf.soil_texture_usda,
    #    clay_av,
    #    silt_av,
    #    dask="parallelized",
    #    output_dtypes=[float],
    #    keep_attrs=True,
    #)

    #soil_texture = soil_texture.raster.reproject_like(ds_like, method="mode")
    #ds_out["soil"] = soil_texture.astype(np.int32)
    
    nodata = -9999.0

    # for writing pcraster map files a scalar nodata value is required
    for var in ds_out:
        ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
        logger.info(f"Interpolate NAN values for {var}")
        ds_out[var] = ds_out[var].fillna(nodata)
        ds_out[var].raster.set_nodata(nodata)
        
    ds_out = ds_out.drop_vars(["SoilThickness","SoilMinThickness", "KsatVer"])

    return ds_out
