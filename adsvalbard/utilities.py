import datetime
import functools
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import projectfiles
import rasterio as rio
import rasterio.coords
import rasterio.transform
import rasterio.warp
import requests
import shapely.geometry

from adsvalbard.constants import CONSTANTS


def get_checksum(objects: list[Any]) -> str:
    return hashlib.sha256("".join(map(str, objects)).encode()).hexdigest()


def align_bounds(
    bounds: rasterio.coords.BoundingBox | dict[str, float],
    res: list[float] | None = None,
    half_mod: bool = False,
    buffer: float | None = None,
) -> rasterio.coords.BoundingBox:
    """
    Align a given bounding box to the given resolution.

    Parameters
    ----------
    bounds
        The input bounding box to the aligned.
    res
        The resolution to align to. If not given, the default is used (constants.py)
    half_mod
        Whether to align the bounding box to half the resolution (see examples)
    buffer
        Buffer the output bounding box with a given distance.

    Examples
    --------
    >>> bounds = rasterio.coords.BoundingBox(0, -0.3, 11, 10)
    >>> align_bounds(bounds, [10., 10.], half_mod=False)
    BoundingBox(left=0.0, bottom=0.0, right=10.0, top=10.0)
    >>> align_bounds(bounds, [10., 10.], half_mod=True)
    BoundingBox(left=-5.0, bottom=5.0, right=5.0, top=15.0)
    >>> align_bounds(bounds, [10., 10.], half_mod=True, buffer=10.)
    BoundingBox(left=-15.0, bottom=-5.0, right=15.0, top=25.0)

    Returns
    -------
    The aligned bounding box.
    """
    if isinstance(bounds, rasterio.coords.BoundingBox):
        bounds_dict: dict[str, float] = {key: getattr(bounds, key) for key in ["left", "bottom", "right", "top"]}
    else:
        bounds_dict = bounds

    if res is None:
        res = [CONSTANTS.res] * 2
    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):

            mod = (bounds_dict[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds_dict[bound] = (
                bounds_dict[bound] - mod + (res[i] if i > 0 and mod != 0 else 0) + ((buffer or 0) * (1 if bound in ["right", "top"] else -1))
            )

    return rasterio.coords.BoundingBox(**bounds_dict)

def get_transform(bounds: rasterio.coords.BoundingBox, res: list[float] | None = None) -> rio.Affine:
    """
    Get the affine transform of the output DEMs.

    Arguments
    ---------
    - res: Optional. Override the x/y resolution.

    Returns
    -------
    An affine transform corresponding to the bounds and the resolution.
    """
    if res is None:
        res = [CONSTANTS.res] * 2

    return rasterio.transform.from_origin(bounds.left, bounds.top, *res)

def get_shape(bounds: rasterio.coords.BoundingBox, res: list[float] | None = None) -> tuple[int, int]:
    """
    Get the pixel shape (height, width) of the output DEMs

    Parameters
    ----------
    - res: Optional. Override the x/y resolution.

    Returns
    -------
    A tuple (height, width) of the output shape.
    """
    # TODO: Remove this in favour of the identical other one!
    if res is None:
        res = [CONSTANTS.res] * 2

    return (int((bounds.top - bounds.bottom) / res[1]), int((bounds.right - bounds.left) / res[0]))


def get_bounds(region: str, res: list[float] | None = None, half_mod: bool = False) -> rasterio.coords.BoundingBox:
    """
    Get the bounding box of the requested region.

    Parameters
    ----------
    region
        The name of the region (from constants.py)
    res
        The resolution for the bounding box to adhere to (see align_bounds()). Defaults to the one in constants.py.
    half_mod
        Whether to align the bounds to half the resolution or the full. See (align_bounds()).

    Returns
    -------
    The calculated bounding box of the region.
    """
    raw_bounds = rasterio.coords.BoundingBox(**CONSTANTS.regions[region])

    return align_bounds(bounds=raw_bounds, res=res, half_mod=half_mod)


def get_crs() -> rio.CRS:
    """Get the target Coordinate Reference System (CRS)."""
    return rio.CRS.from_epsg(CONSTANTS.crs_epsg)


def download_json(url: str) -> dict[str, Any]:
    """
    Download a JSON file and convert it to a dictionary.

    Parameters
    ----------
    url
        The URL of the JSON file

    Returns
    -------
    The JSON parsed as a python dictionary.
    """
    response = requests.get(url=url)
    response.raise_for_status()
    return json.loads(response.content)


def get_data_dir(label: str) -> Path:
    """
    Get the data directory for the given label.

    Parameters
    ----------
    label
        The label of the subdirectory in the main data directory

    Returns
    -------
    The full path to the label.
    """
    data_base_dir = CONSTANTS.data_dir
    if label in ["ArcticDEM", "IC2"]:
        subdir = label
    else:
        raise ValueError(f"Unknown data dir label: {label}")

    return data_base_dir.joinpath(subdir)

        

def download_large_file(url: str, filename: str | None = None, directory: Path | str | None = None) -> Path:
    """
    Download a large file from the given URL.

    If the full target filepath is known:
    `download_large_file(url, filepath.name, filepath.parent)`

    Parameters
    ----------
    url
        The URL to download from.
    filename
        The target filename. If not given, the stem and suffix of the URL is taken.
    directory
        The directory to save the file in. If not given, this defaults to the default cache directory.

    Returns
    -------
    The final path of the downloaded file.
    """
    if isinstance(directory, str) and directory.startswith("/"):
        out_dir = CONSTANTS.cache_dir.joinpath(directory)
    elif isinstance(directory, (str, Path)):
        out_dir = Path(directory)
    else:
        out_dir = CONSTANTS.cache_dir

    if filename is not None:
        out_path = out_dir.joinpath(filename)
    else:
        out_path = out_dir.joinpath(os.path.basename(url))

    if not out_path.is_file():
        with requests.get(url, stream=True) as request:
            request.raise_for_status()

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_path = Path(temp_dir).joinpath("temp.tif")
                with open(temp_path, "wb") as outfile:
                    shutil.copyfileobj(request.raw, outfile)

                shutil.move(temp_path, out_path)

    return out_path

    
    
def shape_from_bounds_res(bounds: rasterio.coords.BoundingBox, res: list[float]) -> tuple[int, int]:
    """
    Get the shape of a raster based on its bounding box and resolution.

    Parameters
    ----------
    bounds
        The bounding box of the raster.
    res
        The resolution [X, Y] of the raster.

    Examples
    --------
    >>> bounds = rasterio.coords.BoundingBox(0, 0, 10, 10)
    >>> shape_from_bounds_res(bounds, [1., 10.])
    (10, 1)
    >>> shape_from_bounds_res(bounds, [1., 1.])
    (10, 10)
            
    Returns
    -------
    The shape (height, width) of the raster.
    """
    return int((bounds.top - bounds.bottom) / res[1]), int((bounds.right- bounds.left) / res[0])


def res_from_bounds_shape(bounds: rasterio.coords.BoundingBox, shape: tuple[int, int]) -> tuple[float, float]:
    """
    Get the resolution of a raster based on its bounding box and shape.

    Parameters
    ----------
    bounds
        The bounding box of the raster.
    shape
        The shape of the raster (height, width)

    Examples
    --------
    >>> bounds = rasterio.coords.BoundingBox(0, 0, 10, 10)
    >>> res_from_bounds_shape(bounds, (10, 10))
    (1.0, 1.0)
    >>> res_from_bounds_shape(bounds, (1, 10))
    (1.0, 10.0)

    Returns
    -------
    The resolution in [X, Y] of the raster.
    """
    return (bounds.right - bounds.left) / shape[1], (bounds.top - bounds.bottom) / shape[0]


def generate_eastings_northings(bounds: rasterio.coords.BoundingBox, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate easting/northing pixel midpoint coordinates for the given bounds and shape.

    Parameters
    ----------
    bounds
        The bounding box of the raster.
    shape
        The shape of the raster (height, width)

    Returns
    -------
    A tuple of (eastings, northings), where eastings.shape == shape. The coordinates represent the center of the pixels.

    """
    res = res_from_bounds_shape(bounds=bounds, shape=shape)

    eastings, northings = np.meshgrid(
        np.linspace(bounds.left +  res[1]/ 2, bounds.right - res[1] / 2, shape[1]),
        np.linspace(bounds.bottom + res[0]/ 2, bounds.top - res[0] / 2, shape[0])[::-1],
    )
    return eastings, northings
    


def warp_bounds(bounds: rasterio.coords.BoundingBox, in_crs: rio.CRS, out_crs: rio.CRS) -> rasterio.coords.BoundingBox:
    """
    Warp the given bounding box from one CRS to another.

    Parameters
    ----------
    bounds
        The bounding box to warp.
    in_crs
        The CRS of the bounding box.
    out_crs
        The target CRS of the output bounding box.

    Returns
    -------
    An equivalent bounding box in the target (`out_crs`) CRS.
    """
    outline = shapely.geometry.box(*bounds).exterior
    points = gpd.GeoSeries([outline.interpolate(i, normalized=True) for i in np.linspace(0, 1)], crs=in_crs).to_crs(out_crs)

    return rasterio.coords.BoundingBox(*points.total_bounds)

    

def bounds_intersect(bounds0: rasterio.coords.BoundingBox, bounds1: rasterio.coords.BoundingBox) -> bool:
    """
    Test whether two bounding boxes intersect or one contains the other.

    Parameters
    ----------
    bounds0
        The first bounding box to evaluate.
    bounds1
        The second bounding box to evaluate.

    Examples
    --------
    >>> bounds0 = rasterio.coords.BoundingBox(0, 0, 1, 1)
    >>> bounds1 = rasterio.coords.BoundingBox(0.3, 0.3, 0.7, 0.7)
    >>> bounds2 = rasterio.coords.BoundingBox(1.1, 1.1, 2., 2.)
    >>> bounds_intersect(bounds0, bounds1)
    True
    >>> bounds_intersect(bounds1, bounds0)
    True
    >>> bounds_intersect(bounds0, bounds2)
    False
    
    Returns
    -------
    A boolean flag whether the boxes intersect or one contains the other.
    """
    box0 = shapely.geometry.box(*bounds0)
    box1 = shapely.geometry.box(*bounds1)
    return box0.overlaps(box1) or box0.within(box1) or box1.within(box0)



def get_resampling_gdal_to_rio() -> dict[str, rasterio.warp.Resampling]:
    """Get a translation dict from GDAL resampling names to rasterio enums."""
    resamplings = {"NearestNeighbour": rasterio.warp.Resampling.nearest, "CubicSpline": rasterio.warp.Resampling.cubic_spline}

    for value in rasterio.warp.Resampling.__dict__:
        if value.startswith("_") or value.endswith("_"):
            continue
        resampling = getattr(rasterio.warp.Resampling, value)
        if resampling in resamplings.values():
            continue

        resamplings[value.capitalize()] = resampling

    return resamplings
    
def resampling_rio_to_gdal(resampling: rasterio.warp.Resampling) -> str:
    """
    Get the GDAL name of the queried rasterio resampling algorithm.

    Parameters
    ----------
    resampling
        The resampling algorithm in rasterio

    Examples
    --------
    >>> resampling_rio_to_gdal(rasterio.warp.Resampling.bilinear)
    'Bilinear'
    >>> resampling_rio_to_gdal(rasterio.warp.Resampling.nearest)
    'NearestNeighbour'
    """
    inverted = {v: k for k, v in get_resampling_gdal_to_rio().items()}
    return inverted[resampling] 


def now_time() -> str:
    """Return a string showing the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")
