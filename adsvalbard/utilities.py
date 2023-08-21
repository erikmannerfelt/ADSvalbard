import geopandas as gpd
import projectfiles
from pathlib import Path
import functools
import rasterio as rio
import json
import numpy as np
from typing import Any
import requests
import os
import tempfile
import shutil
import xarray as xr
import pandas as pd

from adsvalbard.constants import CONSTANTS


def align_bounds(
    bounds: rio.coords.BoundingBox | dict[str, float],
    res: tuple[float, float] | None = None,
    half_mod: bool = True,
    buffer: float | None = None,
) -> rio.coords.BoundingBox:
    if isinstance(bounds, rio.coords.BoundingBox):
        bounds = {key: getattr(bounds, key) for key in ["left", "bottom", "right", "top"]}

    if res is None:
        res = [CONSTANTS.res] * 2
    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):

            mod = (bounds[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds[bound] = (
                bounds[bound] - mod + (res[i] if i > 0 and mod != 0 else 0) + ((buffer or 0) * (1 if i > 0 else -1))
            )

    return rio.coords.BoundingBox(**bounds)

def get_transform(bounds: rio.coords.BoundingBox, res: tuple[float, float] | None = None) -> rio.Affine:
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
        res = CONSTANTS.res

    return rio.transform.from_origin(bounds.left, bounds.top, *res)


def get_shape(bounds: rio.coords.BoundingBox, res: tuple[float, float] | None = None) -> tuple[int, int]:
    """
    Get the pixel shape (height, width) of the output DEMs

    Arguments
    ---------
    - res: Optional. Override the x/y resolution.

    Returns
    -------
    A tuple (height, width) of the output shape.
    """
    if res is None:
        res = [CONSTANTS.res] * 2

    return (int((bounds.top - bounds.bottom) / res[1]), int((bounds.right - bounds.left) / res[0]))


def get_bounds(region: str, res: tuple[float, float] | None = None, half_mod: bool = True) -> rio.coords.BoundingBox:
    raw_bounds = rio.coords.BoundingBox(**CONSTANTS.regions[region])

    return align_bounds(bounds=raw_bounds, res=res, half_mod=half_mod)

def get_crs() -> rio.CRS:
    """Get the target Coordinate Reference System (CRS)."""
    return rio.CRS.from_epsg(CONSTANTS.crs_epsg)


def download_json(url: str) -> dict[str, Any]:
    response = requests.get(url=url)
    response.raise_for_status()
    return json.loads(response.content)


def get_data_dir(label: str) -> Path:
    data_base_dir = CONSTANTS.data_dir
    if label in ["ArcticDEM", "IC2"]:
        subdir = label
    else:
        raise ValueError(f"Unknown data dir label: {label}")

    return data_base_dir.joinpath(subdir)

        

def download_large_file(url: str, filename: str | None = None, directory: Path | str | None = None):

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

    
    
def shape_from_bounds_res(bounds: rio.coords.BoundingBox, res: float) -> tuple[int, int]:
    return int((bounds.top - bounds.bottom) / res[0]), int((bounds.right- bounds.left) / res[1])


def res_from_bounds_shape(bounds: rio.coords.BoundingBox, shape: tuple[int, int]) -> tuple[float, float]:
    return (bounds.top - bounds.bottom) / shape[0], (bounds.right - bounds.left) / shape[1]


