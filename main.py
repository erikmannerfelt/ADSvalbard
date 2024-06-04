from pathlib import Path
from pyproj import CRS
import geopandas as gpd
import rasterio as rio
import rasterio.features
import rasterio.warp
import os
import geoutils as gu
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import re
import json
from tqdm import tqdm
from osgeo import gdal
import datetime
import pandas as pd
import shapely.geometry
import requests
import concurrent.futures
import shutil
import warnings
import tempfile
import threading
import random
import time
import xarray as xr
import scipy.optimize
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.pipeline
from tqdm.dask import TqdmCallback
import skimage
import dask
import dask.distributed
import dask.dataframe as dd
import dask.array as da
import inspect
import numba
import contextlib

import adsvalbard.utilities
from adsvalbard.rasters import build_npi_mosaic, build_stable_terrain_mask

with warnings.catch_warnings():
    warnings.simplefilter("ignore", numba.NumbaDeprecationWarning)
    import xdem

warnings.simplefilter("ignore", RuntimeWarning)
np.seterr(all="ignore")

gdal.UseExceptions()
REGION = "svalbard"

GLACIERS = {
    "dronbreen": {"bounds": [536837.5, 8667472.5, 544677.5, 8679002.5], "region": "nordenskiold"},
    "tinkarp": {"bounds": [548762.5, 8655937.5, 553272.5, 8659162.5], "region": "nordenskiold"},
    "kongsvegen": {"bounds": [445484, 8738240, 468300, 8757503], "region": "haakonvii", "rgi_id": "RGI2000-v7.0-G-07-00254"},
}


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON encoder that properly serializes numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_data_urls():
    np_dem_base_url = "https://public.data.npolar.no/kartdata/S0_Terrengmodell/Delmodell/"

    data_dir = get_data_dir()

    urls = {
        "strip_metadata": [
            (
                "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/strips/s2s041/2m.json",
                "geocell_metadata.json",
            )
        ],
        "NP_DEMs": [
            np_dem_base_url + url
            for url in [
                "NP_S0_DTM5_2008_13652_33.zip",
                "NP_S0_DTM5_2008_13657_35.zip",
                "NP_S0_DTM5_2008_13659_33.zip",
                "NP_S0_DTM5_2008_13660_33.zip",
                "NP_S0_DTM5_2008_13666_35.zip",
                "NP_S0_DTM5_2008_13667_35.zip",
                "NP_S0_DTM5_2009_13822_33.zip",
                "NP_S0_DTM5_2009_13824_33.zip",
                "NP_S0_DTM5_2009_13827_35.zip",
                "NP_S0_DTM5_2009_13833_33.zip",
                "NP_S0_DTM5_2009_13835_33.zip",
                "NP_S0_DTM5_2010_13826_33.zip",
                "NP_S0_DTM5_2010_13828_33.zip",
                "NP_S0_DTM5_2010_13832_35.zip",
                "NP_S0_DTM5_2010_13836_33.zip",
                "NP_S0_DTM5_2010_13918_33.zip",
                "NP_S0_DTM5_2010_13920_35.zip",
                "NP_S0_DTM5_2010_13922_33.zip",
                "NP_S0_DTM5_2010_13923_33.zip",
                "NP_S0_DTM5_2011_13831_35.zip",
                "NP_S0_DTM5_2011_25160_33.zip",
                "NP_S0_DTM5_2011_25161_33.zip",
                "NP_S0_DTM5_2011_25162_33.zip",
                "NP_S0_DTM5_2011_25163_33.zip",
                "NP_S0_DTM5_2012_25235_33.zip",
                "NP_S0_DTM5_2012_25236_35.zip",
                "NP_S0_DTM5_2021_33.zip",
            ]
        ],
        "outlines": [
            "https://public.data.npolar.no/kartdata/NP_S100_SHP.zip",
            (
                "https://api.npolar.no/dataset/"
                + "f6afca5c-6c95-4345-9e52-cfe2f24c7078/_file/3df9512e5a73841b1a23c38cf4e815e3",
                "GAO_SfM_1936_1938.zip",
            ),
        ],
    }

    # TODO: Fix the 35 to 33 issue
    urls["NP_DEMs"] = [url for url in urls["NP_DEMs"] if "_35.zip" not in url]

    for key in urls:
        for i, entry in enumerate(urls[key]):
            if isinstance(entry, tuple):
                filename = entry[1]
                url = entry[0]
            else:
                filename = os.path.basename(entry)
                url = entry

            urls[key][i] = (url, data_dir.joinpath(key).joinpath(filename))

    return urls


def get_strips() -> gpd.GeoDataFrame:

    temp_dir = get_temp_dir()

    output_path = temp_dir.joinpath("strip-meta.feather")

    if output_path.is_file():
        return gpd.read_feather(output_path)

    bounds = get_bounds()
    crs = get_crs()
    bounds_wgs = gpd.GeoSeries(shapely.geometry.box(*list(bounds)), crs=get_crs()).to_crs(4326).total_bounds

    #meta_url, filepath = get_data_urls()["strip_metadata"][0])

    meta_url, filepath = [v for v in get_data_urls()["strip_metadata"] if "geocell_metadata" in str(v[1])][0]

    if not filepath.is_file():
        filepath.parent.mkdir(exist_ok=True, parents=True)
        download_file(meta_url, filepath)

    with open(filepath) as infile:
        data_json = json.load(infile)

    data = pd.DataFrame.from_records(data_json["links"])
    data = data[data["title"].str.contains("Geocell")]
    data["title"] = data["title"].str.replace("Geocell ", "")

    data["lat"] = data["title"].str.slice(1, 3).astype(int)
    data["lon"] = data["title"].str.slice(4, 7).astype(int) * ((data["title"].str.slice(-4, -3) == "e") * 2 - 1)
    data = data[
        (data["lon"] >= np.floor(bounds_wgs[0]))
        & (data["lon"] <= np.ceil(bounds_wgs[2]))
        & (data["lat"] >= np.floor(bounds_wgs[1]))
        & (data["lat"] <= np.ceil(bounds_wgs[3]))
    ]

    cell_files = []
    for _, row in data.iterrows():

        cell_filepath = filepath.parent.joinpath(os.path.basename(row["href"]))
        if not cell_filepath.is_file():
            response = requests.get(row["href"])

            if response.status_code != 200:
                raise ValueError(f"{response.status_code}: {response.content}")

            with open(cell_filepath, "wb") as outfile:
                outfile.write(response.content)

        with open(cell_filepath) as infile:
            files = pd.DataFrame.from_records(json.load(infile)["links"])[["title", "href"]]
            files = files[files["title"].str.contains("SETSM")]
            cell_files.append(files)

    files = pd.concat(cell_files).drop_duplicates("title").sort_values("title")

    strip_metadata_dir = filepath.parent.joinpath("per-strip")
    strip_metadata_dir.mkdir(exist_ok=True)

    infos = []
    filepaths = set()
    for _, row in files.iterrows():
        strip_filepath = strip_metadata_dir.joinpath(os.path.basename(row["href"]))
        filepaths.add(strip_filepath)
        if strip_filepath.is_file():
            continue
        infos.append((strip_filepath, row["href"]))

    filepaths = list(filepaths)

    def download(info, progress_bar=None):
        strip_filepath, url = info

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: {response.content}")

        with open(strip_filepath, "wb") as outfile:
            outfile.write(response.content)

        if progress_bar is not None:
            progress_bar.update()

    if len(infos) > 0:

        # for info in tqdm(infos, desc="Downloading metadata"):
        #    download(info)

        with tqdm(total=len(infos), desc="Downloading metadata") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(lambda i: download(i, progress_bar), infos))

    def load(strip_filepath):

        with open(strip_filepath) as infile:
            data = json.load(infile)

        out = data["properties"]
        del out["description"]
        for i, coord in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            out[f"bbox_{coord}"] = data["bbox"][i]

        out["geometry"] = data["geometry"]
        self_link = [l for l in data["links"] if l["rel"] == "self"][0]
        out["dem"] = os.path.join(os.path.dirname(self_link["href"]), data["assets"]["dem"]["href"][2:])
        out["matchtag"] = os.path.join(os.path.dirname(self_link["href"]), data["assets"]["matchtag"]["href"][2:])

        return out

    strips = gpd.GeoDataFrame.from_records([load(fp) for fp in filepaths])
    strips = strips.set_geometry(strips["geometry"].apply(shapely.geometry.shape), crs=CRS.from_epsg(4326))
    # strips.crs = CRS.from_epsg(4326)
    strips["geometry"] = gpd.GeoSeries.from_wkb(strips["geometry"].to_wkb(output_dimension=2))
    strips = strips.to_crs(crs)

    for col in ["pgc:image_ids", "pgc:avg_sun_elevs", "instruments"]:
        strips[col] = strips[col].apply(lambda col_list: ",".join(map(str, col_list)))

    strips.to_feather(output_path)
    return strips


def download_file(url: str, output_path: Path):

    with requests.get(url, stream=True) as request, tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("file")
        if request.status_code != 200:
            raise ValueError(f"{request.status_code=} {request.content=}")

        with open(temp_file, "wb") as outfile:
            shutil.copyfileobj(request.raw, outfile)

        shutil.move(temp_file, output_path)


def download_arcticdem(dem_data: pd.Series, verbose: bool = True):
    data_dir = get_data_dir()
    ad_dir = data_dir.joinpath("ArcticDEM")

    filepaths = []
    for kind in ["dem", "matchtag"]:
        filepath = ad_dir.joinpath(os.path.basename(dem_data[kind]))
        if not filepath.is_file():
            if verbose:
                print(f"{now_time()}: Downloading {dem_data[kind]}")
            with requests.get(dem_data[kind], stream=True, timeout=30) as request:
                request.raise_for_status()
                with tempfile.TemporaryDirectory() as temp_dir:

                    temp_path = Path(temp_dir).joinpath("temp.tif")
                    with open(temp_path, "wb") as outfile:
                        shutil.copyfileobj(request.raw, outfile)

                    shutil.move(temp_path, filepath)

        filepaths.append(filepath)

    return filepaths[0], filepaths[1]


def get_xdem_dem_co() -> dict[str, str]:
    """Get DEM creation options for xdem."""
    return {"COMPRESS": "DEFLATE", "ZLEVEL": "12", "TILED": "YES", "NUM_THREADS": "ALL_CPUS"}


def get_dem_co() -> list[str]:
    """Get DEM creation options for GDAL."""
    return [f"{k}={v}" for k, v in get_xdem_dem_co().items()]


def get_res() -> tuple[float, float]:
    """Get the horizontal/vertical resolution of the output DEMs."""
    return (5.0, 5.0)


def align_bounds(
    bounds: rio.coords.BoundingBox | dict[str, float],
    res: tuple[float, float] | None = None,
    half_mod: bool = False,
    buffer: float | None = None,
) -> rio.coords.BoundingBox:
    if isinstance(bounds, rio.coords.BoundingBox):
        bounds = {key: getattr(bounds, key) for key in ["left", "bottom", "right", "top"]}

    if res is None:
        res = get_res()
    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):

            mod = (bounds[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds[bound] = (
                bounds[bound] - mod + (res[i] if i > 0 and mod != 0 else 0) + ((buffer or 0) * (1 if i > 0 else -1))
            )

    return rio.coords.BoundingBox(**bounds)


def get_bounds(
    region: str = REGION, res: tuple[float, float] | None = None, half_mod: bool = False
) -> rio.coords.BoundingBox:
    """
    Get the bounding coordinates of the output DEMs.

    Arguments
    ---------
    - region: The selected region (with hardcoded bounds)
    - res: Optional. Override the x/y resolution.
    - half_mod: Whether the modulus should be calculated on half the pixel size (e.g. 50000 at 5m -> 50002.5)

    Returns
    -------
    A bounding box of the region.
    """

    if region != "svalbard":

        region_bounds = {
            "svalbard": {"left": 341002.5, "bottom": 8455002.5, "right": 905002.5, "top": 8982002.5},
            "nordenskiold": {"left": 443002.5, "bottom": 8626007.5, "right": 560242.5, "top": 8703007.5},
            "heerland": {"left": 537010, "bottom": 8602780, "right": 582940, "top": 8656400},
            "kongsvegen": {"left": 438960, "bottom": 8735190, "right": 470420, "top": 8764230},
            "haakonvii": {"left": 412970, "bottom": 8689380, "right": 495820, "top": 8837790}
        }

        bounds = region_bounds[region]

        return align_bounds(bounds, res=res, half_mod=half_mod)

    crs = get_crs()
    roi_path = Path("shapes/region_of_interest.geojson")
    roi = gpd.read_file(roi_path).to_crs(crs)
    res = get_res()
    return adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*roi.total_bounds), res=res, half_mod=False)



def get_transform(res: tuple[float, float] | None = None) -> rio.Affine:
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
        res = get_res()

    bounds = get_bounds(res=res)
    return rio.transform.from_origin(bounds.left, bounds.top, *res)


def get_shape(res: tuple[float, float] | None = None) -> tuple[int, int]:
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
        res = get_res()

    bounds = get_bounds(res=res)

    return (int((bounds.top - bounds.bottom) / res[1]), int((bounds.right - bounds.left) / res[0]))


def get_crs() -> CRS:
    """Get the target Coordinate Reference System (CRS)."""
    return CRS.from_epsg(32633)


def get_temp_dir(label: str = REGION) -> Path:
    """Get (and create) the path to the directory to store temporary files in."""

    temp_dir = get_data_dir().parent.joinpath(f"temp.{label}")
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path("data/").absolute()


def build_stable_terrain_mask_old(region: str = REGION, verbose: bool = False) -> Path:
    """
    Build a stable terrain mask.

    The mask excludes areas within:

    - 1936 glacier outlines from Geyman et al., 2022
    - Water (rivers/lakes) according to the NPI S100 map
    - Moraines according to the NPI S100 map

    ... and includes all other areas on land according to the NPI S100 map

    Arguments
    ---------
    - verbose: Whether to print updates to the console

    Returns
    -------
    A path to the stable terrain mask.
    """

    data_dir = get_data_dir()
    temp_dir = get_temp_dir(label=region)
    out_filepath = temp_dir.joinpath("stable_terrain.tif")

    if out_filepath.is_file():
        return out_filepath

    res = get_res()
    shape = get_shape(res=res)
    transform = get_transform(res=res)
    crs = get_crs()

    geyman_1936 = (
        gpd.read_file(f"zip://{data_dir}/GAO_SfM_1936_1938.zip!GAO_SfM_1936_1938_v3.shp")
        .buffer(0)
        .to_frame()
        .dissolve()
        .to_crs(crs)
    )
    water = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Vann_f.shp").dissolve().to_crs(crs)
    moraines = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Morener_f.shp").dissolve().to_crs(crs)
    land = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Land_f.shp").dissolve().to_crs(crs)

    stable = land.difference(geyman_1936.union(water).union(moraines))
    if verbose:
        print(f"{now_time()}: Loaded stable terrain features in memory")

    rasterized = rasterio.features.rasterize(stable.geometry, out_shape=shape, transform=transform, default_value=1)
    if verbose:
        print(f"{now_time()}: Rasterized features.")

    os.makedirs(out_filepath.parent, exist_ok=True)
    with rio.open(
        out_filepath,
        "w",
        crs=crs,
        transform=transform,
        compress="lzw",
        tiled=True,
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
    ) as raster:
        raster.write(rasterized, 1)
    if verbose:
        print(f"{now_time()}: Saved {out_filepath.name}.")

    return out_filepath


def create_warped_vrt(
    filepath: Path | str, vrt_filepath: Path | str, out_crs: str | CRS, in_crs: str | CRS | None = None
) -> None:
    """
    Create a warped VRT from a raster with a different CRS.

    :param filepath: The path to the raster to create a VRT from.
    :param vrt_filepath: The output path of the VRT.
    :param out_crs: The target CRS of the VRT either as a WKT or a CRS object.
    :param in_crs: The original CRS of the VRT either as a WKT or a CRS object.
    """
    ds = gdal.Open(str(filepath))

    if isinstance(out_crs, CRS):
        out_crs = out_crs.to_wkt()
    if isinstance(in_crs, CRS):
        in_crs = in_crs.to_wkt()

    vrt = gdal.AutoCreateWarpedVRT(ds, in_crs, out_crs, rasterio.warp.Resampling.bilinear)
    vrt.GetDriver().CreateCopy(str(vrt_filepath), vrt)

    del ds
    del vrt



def build_npi_mosaic_old(region: str = REGION, verbose: bool = False) -> tuple[Path, Path]:
    """
    Build a mosaic of tiles downloaded from the NPI.

    Arguments
    ---------
    - verbose: Whether to print updates to the console

    Returns
    -------
    A path to the NPI mosaic.
    """

    data_dir = get_data_dir()
    temp_dir = get_temp_dir(label=region)

    output_path = temp_dir.joinpath("npi_mosaic_clip.tif")
    output_year_path = temp_dir.joinpath("npi_mosaic_clip_years.tif")

    if output_path.is_file() and output_year_path.is_file():
        return output_path, output_year_path

    from osgeo import gdal

    crs = get_crs()
    res = get_res()
    bounds = get_bounds(res=res)

    bounds_epsg25833 = shapely.geometry.box(*gpd.GeoSeries([shapely.geometry.box(*bounds)], crs=32633).to_crs(25833).buffer(50).total_bounds)

    # Generate links to the DEM tiles within their zipfiles
    uris = []
    year_rasters = []
    for url, filepath in tqdm(get_data_urls()["NP_DEMs"], desc="Preparing DEMs and year info", disable=(not verbose)):
        year = int(filepath.stem.split("_")[3])

        if not filepath.is_file():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, filepath)

        uri = f"/vsizip/{filepath}/{filepath.stem}/{filepath.stem.replace('NP_', '')}.tif"

        dem = xdem.DEM(uri, load_data=False)

        dem_bbox = shapely.geometry.box(*dem.bounds)
        if not bounds_epsg25833.overlaps(dem_bbox):
            # TODO: Figure out why 13822 doesn't work with Kongsvegen...
            if inspect.signature(get_bounds).parameters["region"].default in ["kongsvegen", "haakonvii"] and "13822" in filepath.stem:
                pass
            elif inspect.signature(get_bounds).parameters["region"].default in ["haakonvii"] and "13828" in filepath.stem:
                pass
            else:
                continue

        dem.crop(bounds_epsg25833.bounds)
       
        year_raster = gu.Raster.from_array(
            (np.zeros(dem.shape, dtype="uint16") + year) * (1 - dem.data.mask.astype("uint16")),
            transform=dem.transform,
            crs=dem.crs,
            nodata=0,
        )

        year_rasters.append(year_raster)
        uris.append(uri)

    if verbose:
        print("Merging rasters")
    year_raster = gu.raster.merge_rasters(year_rasters, merge_algorithm=np.nanmax, resampling_method="nearest")

    year_raster.reproject(dst_bounds=bounds, dst_res=res, dst_crs=crs, resampling="nearest").save(
        output_year_path, tiled=True, compress="lzw"
    )
    del year_raster

    vrt_dir = temp_dir.joinpath("npi_vrts/")
    vrt_dir.mkdir(exist_ok=True)
    mosaic_path = vrt_dir.joinpath("npi_mosaic.vrt")

    # Mosaic the tiles in a VRT
    gdal.BuildVRT(str(mosaic_path), uris)

    if verbose:
        print("Saving DEM mosaic")

    # Warp the VRT into one TIF
    gdal.Warp(
        str(output_path),
        str(mosaic_path),
        dstSRS=crs.to_wkt(),
        creationOptions=get_dem_co(),
        xRes=res[0],
        yRes=res[1],
        resampleAlg=rasterio.warp.Resampling.cubic_spline,
        outputBounds=list(bounds),
        multithread=True,
    )
    return output_path, output_year_path


def prepare_mask(filepath: Path) -> np.ndarray:
    """
    Load and modify the mask of an ArcticDEM to be more conservative.

    1. Dilate the excluded regions to exclude the periphery and remove lone pixels.
    2. Remove all disconnected patches with an area of less than 1% of the largest patch.

    Arguments
    ---------
    - filepath: The path to the mask to load and modify

    Returns
    -------
    A boolean outlier mask (True if a pixel should be excluded) array

    """
    outlier_mask = gu.Raster(str(filepath)).data.filled(0) == 0

    if np.count_nonzero(~outlier_mask) == 0:
        raise ValueError("Mask is empty")

    # Dilate the mask (expand the excluded areas)
    struct = scipy.ndimage.iterate_structure(scipy.ndimage.generate_binary_structure(2, 1), 3)
    outlier_mask = scipy.ndimage.binary_dilation(outlier_mask.squeeze(), struct)

    # Label each connected inlier component
    labelled, _ = scipy.ndimage.label(~outlier_mask)
    # Count the numbers of unique labels
    unique, counts = np.unique(labelled, return_counts=True)
    # Zero is assumed to be the label of the outliers (which we want to exclude from this part)
    counts = counts[unique != 0]
    unique = unique[unique != 0]
    # Create a new mask which is False (inliers) for all patches that are larger than 1% of the largest patch
    # All small patches and the previous outliers are True (outliers)
    outlier_mask = np.isin(labelled, unique[counts > int(counts[np.argmax(counts)] * 0.01)], invert=True)

    return outlier_mask


def prepare_arcticdem(dem_path: Path) -> tuple[Path, Path]:
    """
    Prepare warped and cropped VRTs for one ArcticDEM.

    Arguments
    ---------
    - dem_path: The filepath to the ArcticDEM

    Returns
    -------
    A tuple containing a warped VRT of the DEM and its mask (DEM, mask).
    """
    temp_dir = get_temp_dir()
    crs = get_crs()
    res = get_res()
    full_bounds = get_bounds(res=res)

    ad_vrt_dir = temp_dir.joinpath("arcticdem_vrts")
    ad_vrt_dir.mkdir(exist_ok=True)

    mask_path = dem_path.with_stem(dem_path.stem.replace("_dem", "_matchtag"))

    vrts = []

    bounds: rio.coords.BoundingBox | None = None

    for i, filepath in enumerate([dem_path, mask_path]):
        # First create a warped VRT to convert the CRS
        warp_vrt_path = ad_vrt_dir.joinpath(filepath.stem + f"_epsg{crs.to_epsg()}.vrt")

        if i == 0:
            with rio.open(filepath) as raster:

                # Read the mask of overview level 8 if it exists. Otherwise the second overview, or worst, no overview
                overviews = raster.overviews(1)
                overview_level = 8 if 8 in overviews else overviews.get(2, 1)
                mask = raster.read_masks(1, out_shape=(int(raster.height / overview_level), int(raster.width / overview_level)))

                # Generate approximate coordinates of the pixels in the wrong CRS
                coords = np.array(np.meshgrid(
                    np.linspace(raster.bounds.left, raster.bounds.right, mask.shape[1]),
                    np.linspace(raster.bounds.bottom, raster.bounds.top, mask.shape[0])[::-1],
                ))

                # Convert the coordinates to the right CRS and mask them to all data pixels only
                coords = gpd.points_from_xy(*coords.reshape((2, -1)).T[mask.ravel() > 0].T, crs=raster.crs).to_crs(crs)

                # Find the smallest "right CRS" bounds of the unmasked pixels in the raster
                warp_bounds = align_bounds(rio.coords.BoundingBox(*coords.total_bounds), res=res, buffer=res[0])

            # Intersect the smallest available bounds with the total bounds of the domain
            bounds = rio.coords.BoundingBox(
                left=max(full_bounds.left, warp_bounds.left),
                bottom=max(full_bounds.bottom, warp_bounds.bottom),
                right=min(full_bounds.right, warp_bounds.right),
                top=min(full_bounds.top, warp_bounds.top),
            )

        # First warp the DEM to the right CRS
        create_warped_vrt(str(filepath), str(warp_vrt_path), out_crs=crs)

        # This is just to make the linters happy. It should never be true
        if bounds is None:
            raise NotImplementedError()

        # Then, build a new VRT to resample and crop the raster
        out_path = warp_vrt_path.with_stem(warp_vrt_path.stem + f"_{res[0]}m")
        gdal.BuildVRT(
            str(out_path),
            str(warp_vrt_path),
            outputBounds=list(bounds),
            xRes=res[0],
            yRes=res[1],
            resampleAlg="bilinear",
        )
        vrts.append(out_path)

    return vrts[0], vrts[1]


def now_time() -> str:
    """Return a string showing the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def coregister(dem_path: Path, verbose: bool = True):
    """
    Co-register an ArcticDEM to the NPI mosaic.

    The used approach is ICP for large translation/rotation and then NuthKaab for sub-pixel alignment.
    The stable terrain mask is used to exclude potential unstable terrain.
    Co-registration parameters are saved alongside the co-registered DEM.

    Arguments
    ---------
    - dem_path: The filepath to the ArcticDEM
    - verbose: Whether to print updates to the console

    Returns
    -------
    The path to the co-registered ArcticDEM
    """
    temp_dir = get_temp_dir()
    coreg_dir = temp_dir.joinpath("arcticdem_coreg")

    output_path = coreg_dir.joinpath(dem_path.stem + "_coreg.tif")
    if output_path.is_file():
        return output_path

    coreg_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"{now_time()}: Processing {dem_path.name}")

    dem_vrt_path, mask_vrt_path = prepare_arcticdem(dem_path=dem_path)

    mask = prepare_mask(mask_vrt_path)
    if verbose:
        print(f"\t{now_time()}: Loaded and modified mask")

    tba_dem: gu.Raster = gu.Raster(str(dem_vrt_path), load_data=True)
    tba_dem.set_mask(mask)

    if np.count_nonzero(np.isfinite(tba_dem.data.filled(np.nan))) == 0:
        raise ValueError("No finite values in TBA DEM")
    if verbose:
        print(f"\t{now_time()}: Loaded TBA DEM")

    stable_terrain_path = build_stable_terrain_mask(verbose=verbose)
    stable_terrain_mask = gu.Raster(str(stable_terrain_path), load_data=False).crop(tba_dem, inplace=False)
    stable_terrain_mask = stable_terrain_mask.data.filled(0) == 1

    if np.count_nonzero(stable_terrain_mask) == 0:
        raise ValueError("No stable terrain pixels in window")
    if np.count_nonzero(np.isfinite(tba_dem.data.filled(np.nan)[stable_terrain_mask])) == 0:
        raise ValueError("No overlapping stable terrain")
    if verbose:
        print(f"\t{now_time()}: Loaded stable terrain mask")

    del mask

    npi_mosaic_path, _ = build_npi_mosaic(verbose=verbose)
    ref_dem = gu.Raster(str(npi_mosaic_path), load_data=False).crop(tba_dem, inplace=False)
    if verbose:
        print(f"\t{now_time()}: Loaded ref. DEM. Running co-registration")
    if np.count_nonzero(np.isfinite(ref_dem.data.filled(np.nan))) == 0:
        raise ValueError("No finite values in reference DEM")

    # if np.count_nonzero(stable_terrain_mask) == 0 or np.count_nonzero(np.isfinite(ref_dem.data.filled(np.nan))) == 0
    coreg = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
    coreg.fit(
        reference_dem=ref_dem.data,
        dem_to_be_aligned=tba_dem.data,
        transform=ref_dem.transform,
        crs=tba_dem.crs,
        inlier_mask=stable_terrain_mask,
    )

    out_meta = output_path.with_suffix(".json")
    metadata: dict[str, object] = {"steps": []}
    for part in coreg.pipeline:
        metadata["steps"].append(
            {
                "name": re.search(r"coreg.*'", str(part.__class__)).group().replace("'", "").replace("coreg.", ""),
                "meta": part.__dict__,
            }
        )

    if verbose:
        print(f"\t{now_time()}: Finished co-registration. Transforming DEM")
    tba_dem = coreg.apply(tba_dem)

    diff = tba_dem.data[stable_terrain_mask].filled(np.nan) - ref_dem.data[stable_terrain_mask].filled(np.nan)
    metadata["stable_nmad"] = float(xdem.spatialstats.nmad(diff))
    metadata["stable_bias"] = float(np.nanmedian(diff))
    metadata["stable_fraction"] = float(np.count_nonzero(stable_terrain_mask) / np.prod(stable_terrain_mask.shape))
    metadata["pixel_count"] = int(np.count_nonzero(np.isfinite(tba_dem.data)))
    metadata["bounds"] = {b: float(getattr(tba_dem.bounds, b)) for b in ["left", "bottom", "right", "top"]}
    metadata["crs"] = f"EPSG:{tba_dem.crs.to_epsg()}"

    with open(out_meta, "w") as outfile:
        json.dump(metadata, outfile, cls=NumpyArrayEncoder)

    if verbose:
        print(f"\t{now_time()}: Applied co-registration")

    tba_dem.save(str(output_path), co_opts=get_xdem_dem_co())

    if verbose:
        print(f"\t{now_time()}: Saved {output_path.name}")

    return output_path


def generate_difference(dem_path: Path, verbose: bool = False):

    date = pd.to_datetime(dem_path.stem.split("_")[3], format="%Y%m%d")
    temp_dir = get_temp_dir()
    dh_dir = temp_dir.joinpath(f"dh/{date.year}")
    dhdt_dir = temp_dir.joinpath(f"dhdt/{date.year}")
    dt_dir = temp_dir.joinpath(f"dt/{date.year}")

    output_dh_path = dh_dir.joinpath(dem_path.stem + "_dh.tif")
    output_dhdt_path = dhdt_dir.joinpath(dem_path.stem + "_dhdt.tif")
    output_dt_path = dt_dir.joinpath(dem_path.stem + "_dt.tif")
    if all(fp.is_file() for fp in [output_dh_path, output_dhdt_path, output_dt_path]):
        return output_dh_path, output_dhdt_path

    for directory in [dh_dir, dt_dir, dhdt_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    dem = xdem.DEM(str(dem_path), read_from_fn=False)

    npi_mosaic_path, npi_mosaic_years_path = build_npi_mosaic(verbose=verbose)
    npi_dem = xdem.DEM(str(npi_mosaic_path), load_data=False).crop(dem, inplace=False)
    npi_dem_years = xdem.DEM(str(npi_mosaic_years_path), load_data=False).crop(dem, inplace=False).astype("float32")
    # Assume that it's the 1st of August of every year
    npi_dem_years += 8 / 12

    date = pd.to_datetime(dem_path.stem.split("_")[3], format="%Y%m%d")
    year_decimal = date.year + date.month / 12 + date.day / (30 * 12)

    dh_map = dem - npi_dem

    dt_map = year_decimal - npi_dem_years
    dt_map.data.mask = dh_map.data.mask
    dhdt_map = dh_map / dt_map
    dt_map.save(output_dt_path, co_opts=get_xdem_dem_co())
    dhdt_map.save(output_dhdt_path, co_opts=get_xdem_dem_co())
    dh_map.save(output_dh_path, co_opts=get_xdem_dem_co(), nodata=-9999)

    if verbose:
        print(f"{now_time()}: Saved dH/dt map")

    return output_dh_path, output_dhdt_path


def median_stack():

    temp_dir = get_temp_dir()
    dh_dir = temp_dir.joinpath("dh")

    bounds = rio.coords.BoundingBox(535177.5, 8666322.5, 547662.5, 8678382.5)

    start_date = pd.Timestamp("2009-07-25")

    output_data = {}
    dhs = []

    filepaths = list(dh_dir.glob("*_dh.tif"))
    for filepath in tqdm(filepaths):

        date = pd.to_datetime(filepath.stem.split("_")[3], format="%Y%m%d")

        # if date.year < 2020:
        #    continue

        year_diff = (date - start_date).total_seconds() / (3600 * 24 * 365.24)

        with rio.open(filepath) as raster:
            window = rio.windows.from_bounds(*bounds, transform=raster.transform)
            output_data["transform"] = rio.windows.transform(window, raster.transform)
            output_data["crs"] = raster.crs

            dhs.append(
                np.clip(
                    raster.read(1, window=window, masked=True, boundless=True, fill_value=-9999).filled(np.nan)
                    / year_diff,
                    -4,
                    4,
                )
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(dhs, axis=0)

    plt.imshow(median, cmap="coolwarm_r", vmin=-3, vmax=3)

    with rio.open(
        temp_dir.joinpath("median.tif"),
        "w",
        driver="GTiff",
        width=median.shape[1],
        height=median.shape[0],
        count=1,
        crs=output_data["crs"],
        transform=output_data["transform"],
        dtype=median.dtype,
        nodata=-9999,
        compress="deflate",
        tiled=True,
    ) as raster:
        raster.write(np.where(np.isfinite(median), median, -9999), 1)
    plt.show()


def big_median_stack(years: int | list[int] | None = 2021, n_threads: int | None = None, raster_type: str = "dhdt", verbose: bool = True, write_in_mem: bool = False):
    temp_dir = get_temp_dir()

    if raster_type == "dhdt":
        raster_dir = temp_dir.joinpath("dhdt")
    elif raster_type == "dem":
        raster_dir = temp_dir.joinpath("arcticdem_coreg")
        
    else:
        raise NotImplementedError(f"Unknown raster type: {raster_type}") 


    if years is None:
        ext = ""
        dirs = [d for d in raster_dir.glob("*") if d.is_dir()]
    if isinstance(years, int):
        ext = "_" + str(years)
        dirs = [raster_dir.joinpath(str(years))]
        years = [years]
    elif isinstance(years, list):
        ext = "_" + "_".join(map(str, years))
        dirs = [raster_dir.joinpath(str(year)) for year in years]
    else:
        raise TypeError(f"{years=} has unknown type: {type(years)=}")

    if raster_type == "dhdt":
        output_path = raster_dir.joinpath(f"median_dhdt{ext}.tif")
        v_clip = 150 / (2021 - 2009)
        pattern = "*_dhdt.tif"
        # raster_files = []
        # for directory in dirs:
        #     raster_files += list(directory.glob(pattern))
    elif raster_type == "dem":
        output_path = temp_dir.joinpath(f"median_dem{ext}.tif")
        v_clip = 1500
        pattern = "*_dem_coreg.tif"
    else:
        raise NotImplementedError(f"Unknown raster type: {raster_type}") 
        
    raster_files = list(filter(lambda fp: int(fp.stem.split("_")[3][:4]) in years, raster_dir.rglob(pattern)))

    if output_path.is_file():
        return output_path

    res = get_res()
    shape = get_shape(res=res)
    bounds = get_bounds(res=res)
    transform = get_transform(res=res)
    crs = get_crs()

    block_size = [512] * 2

    strips = get_strips()


    titles = {r_path.stem[: r_path.stem.index("_seg") + 5]: r_path for r_path in raster_files}

    locks = {path: threading.Lock() for path in titles.values()}

    write_lock = threading.Lock()

    window_infos = []
    for col_off in np.arange(0, shape[1], step=block_size[0]):
        for row_off in np.arange(0, shape[0], step=block_size[1]):
            width = min(shape[1] - col_off, block_size[0])
            height = min(shape[0] - row_off, block_size[1])

            window = rio.windows.Window(col_off, row_off, width, height)

            win_bounds = rio.windows.bounds(window, transform=transform)

            overlapping_strips = strips[strips.intersects(shapely.geometry.box(*win_bounds))]

            paths = []
            for title in overlapping_strips["title"].values:
                if title not in titles:
                    continue
                paths.append((titles[title], locks[titles[title]]))

            if len(paths) == 0:
                continue

            window_infos.append({"window": window, "paths": paths})

    # Shuffle the windows to reduce the chance that multiple threads wait to read the same file
    random.shuffle(window_infos)

    write_params = dict(
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=1,
        crs=crs,
        transform=transform,
        dtype="float32",
        nodata=-9999,
        compress="deflate",
        tiled=True,
        zlevel=12,
    )

    with contextlib.ExitStack() as stack:
        
        if write_in_mem:
            stack = np.zeros(shape, dtype="float32") - 9999
        else:
            out_raster = stack.enter_context(rio.open(output_path, "w", **write_params))
            

        def process(
            window_info: list[dict[str, rio.windows.Window | tuple[Path, threading.Lock]]], progress_bar: tqdm | None = None
        ) -> None:

            window: rio.windows.Window = window_info["window"]
            window_bounds = rio.windows.bounds(window, transform)

            data = []

            for path, lock in window_info["paths"]:
                with lock:
                    with rio.open(path) as raster:
                        raster_win = rio.windows.from_bounds(*window_bounds, raster.transform)

                        data.append(
                            np.clip(
                                raster.read(1, window=raster_win, masked=True, boundless=True).filled(np.nan),
                                -v_clip,
                                v_clip,
                            )
                        )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice")
                median = np.nanmedian(data, axis=0)
            del data

            if np.count_nonzero(np.isfinite(median)) == 0:
                progress_bar.update()
                return median

            # median[~np.isfinite(median)] = -9999

            with write_lock:
                # print(window.row_off, window.row_off + window.height, window.col_off,window.col_off + window.width)
                if write_in_mem:
                    stack[
                        window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width
                    ] = median
                else:
                    out_raster.write(np.where(np.isfinite(median), median, -9999), 1, window=window)

            progress_bar.update()
            return median

        with tqdm(total=len(window_infos), desc="Calculating median blocks", smoothing=0.1) as progress_bar:
            if n_threads == 1:
                for window_info in window_infos:
                    process(window_info=window_info, progress_bar=progress_bar)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "All-NaN slice")
                        list(executor.map(lambda wi: process(wi, progress_bar=progress_bar), window_infos))

    if write_in_mem:
        if verbose:
            print(f"{now_time()}: Writing {output_path.name}")
        with rio.open(
            output_path,
            "w",
            **write_params
        ) as raster:
            raster.write(np.where(np.isfinite(stack), stack, -9999), 1)

    return output_path

    # windows = np.ravel([[rio.windows.Window(col_off, row_off, min(shape[1] - col_off, block_size[0]), min(shape[0] - row_off, block_size[1])) for col_off in np.arange(0, shape[1], step=block_size[0])] for row_off in np.arange(0, shape[0], step=block_size[1])])


def uncertainty():

    poi_coords = 538280, 8669555, 544315, 8675410
    poi = shapely.geometry.box(*poi_coords)
    stable_terrain_path = build_stable_terrain_mask()

    median_stack_path = Path("temp/dhdt/median_dhdt_2021.tif")
    npi_mosaic_path, _ = build_npi_mosaic()

    npi_dem = xdem.DEM(str(npi_mosaic_path), load_data=False).crop(poi_coords, inplace=False)
    stable_terrain = gu.Raster(str(stable_terrain_path), load_data=False).crop(poi_coords, inplace=False)

    unstable_terrain_mask = stable_terrain.data != 1

    median = gu.Raster(str(median_stack_path), load_data=False).crop(poi_coords, inplace=False)

    # median.data.mask[stable_terrain.data != 1] = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Setting default nodata")
        slope, maxc = xdem.terrain.get_terrain_attribute(npi_dem, ["slope", "maximum_curvature"])

    errors, df_binning, error_function = xdem.spatialstats.infer_heteroscedasticity_from_stable(
        dvalues=median, list_var=[slope, maxc], list_var_names=["slope", "maxc"], unstable_mask=unstable_terrain_mask
    )

    zscores = median / errors
    (
        emp_variogram,
        params_variogram_model,
        spatial_corr_function,
    ) = xdem.spatialstats.infer_spatial_correlation_from_stable(
        dvalues=zscores, list_models=["Gaussian", "Spherical"], unstable_mask=unstable_terrain_mask, random_state=42
    )

    areas = 10 ** np.linspace(1, 12)
    stderrs = xdem.spatialstats.spatial_error_propagation(
        areas=areas, errors=errors, params_variogram_model=params_variogram_model
    )

    plt.plot(areas / 10**6, stderrs)
    plt.xlabel("Averaging area (km²)")
    plt.ylabel("Standard error (m/a)")
    plt.xscale("log")

    plt.savefig("temp/errors.jpg", dpi=300)

    df = xdem.spatialstats.sample_empirical_variogram(
        values=np.where(unstable_terrain_mask, np.nan, zscores.data.filled(np.nan)),
        gsd=median.res[0],
        subsample=100,
        n_variograms=10,
        random_state=42,
    )
    # Standardize by the error so the y-axis makes sense
    df[["exp", "err_exp"]] *= errors.data.mean()
    print(df)

    xdem.spatialstats.plot_variogram(df, xscale="log")
    plt.savefig("temp/variogram.jpg", dpi=300)

    # slope.save("temp/npi_slope.tif")
    # maxc.save("temp


def process_all(show_progress_bar: bool = True):

    strips = get_strips()

    # All of Nordenskiöld Land
    poi = shapely.geometry.box(*get_bounds(half_mod=False))

    # Build these first (or conversely, check that they exist)
    build_npi_mosaic(verbose=(not show_progress_bar))
    build_stable_terrain_mask(verbose=(not show_progress_bar))
    # Drønbreen
    # poi = shapely.geometry.box(538286, 8669555,544315,8675416)
    # Tinkarpbreen
    # poi = shapely.geometry.box(548766,8655934,553271,8659162)

    # Remove the northern part of Isfjorden for Nordenskiold because it's too small to use.
    if REGION == "nordenskiold":
        poi = poi.difference(shapely.geometry.box(435428, 8679301, 497701, 8714978))

    failures_file = Path(f"failures.{REGION}.csv")

    strips = strips[strips.intersects(poi)].sort_values("start_datetime", ascending=False)
    if failures_file.is_file():
        failures = pd.read_csv(failures_file, names=["title", "exception"])
        strips = strips[~strips["title"].isin(failures["title"])]

    strips["start_datetime"] = pd.to_datetime(strips["start_datetime"])

    current_year = strips["start_datetime"].dt.year.max()  # type: ignore
    with tqdm(total=strips.shape[0], disable=(not show_progress_bar)) as progress_bar:
        for _, dem_data in strips.iterrows():

            if dem_data["start_datetime"].year != current_year:  # type: ignore
                if not show_progress_bar:
                    print(f"{now_time()}: Finished {current_year}. Creating dHdt and DEM mosaics")
                else:
                    progress_bar.set_description(f"Finished {current_year}. Creating dHdt and DEM mosaics")
                for raster_type in ["dhdt", "dem"]:
                    big_median_stack(years=current_year, raster_type=raster_type, verbose=(not show_progress_bar))

                current_year = dem_data["start_datetime"].year  # type: ignore

            start_time = time.time()
            if show_progress_bar:
                progress_bar.set_description(f"Working on {dem_data['title']}")
            dem_path, _ = download_arcticdem(dem_data, verbose=(not show_progress_bar))
            try:
                dem_coreg = coregister(dem_path=dem_path, verbose=(not show_progress_bar))
            except Exception as exception:

                exception_str = str(exception)
                if exception.__class__.__name__ == "AssertionError" and len(exception_str) == 0:
                    with open(failures_file, "a+") as outfile:
                        outfile.write(dem_data["title"] + ',"Empty AssertionError; probably xdem"\n')
                    progress_bar.update()
                    if (not show_progress_bar):
                        print(f"{now_time()}: {dem_data['title']} failed. Recorded failure")
                    continue

                if not any(
                    s in exception_str
                    for s in [
                        "Transformed DEM has all nans",
                        "ICP coregistration failed",
                        "Not enough valid points",
                        "Mask is empty",
                        "No finite values in",
                        "No stable terrain pixels in window",
                        "Expected one value, found 0",
                        "No overlapping stable terrain",
                        "Less than 10 different cells exist",
                        "Intersection is empty",
                    ]
                ):

                    raise exception
                with open(failures_file, "a+") as outfile:
                    outfile.write(dem_data["title"] + ',"' + str(exception).replace("\n", " ") + '"\n')
                if (not show_progress_bar):
                    print(f"{now_time()}: {dem_data['title']} failed. Recorded failure")
                progress_bar.update()
                continue
            generate_difference(dem_coreg, verbose=(not show_progress_bar))

            # If it was really fast, all files already existed and were only validated
            if (time.time() - start_time) < 3:
                progress_bar.total = progress_bar.total - 1
                progress_bar.refresh()
            else:
                progress_bar.update()


def ransac_fit(heights: da.Array | np.ndarray, time: da.Array | np.ndarray, degree: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inlier_mask = np.isfinite(heights)
        if heights.shape[0] == 1 or (np.count_nonzero(inlier_mask) < 2):
            return [np.nan] * (degree + 1)

        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(degree=degree),
            sklearn.linear_model.RANSACRegressor(min_samples=2, max_trials=100, random_state=1),
        )
        model.fit(
            time[inlier_mask].reshape((-1, 1)),
            heights[inlier_mask],
            # ransacregressor__sample_weight=points["weight"].values,
        )
        estimator = model.steps[-1][1].estimator_

        return np.r_[estimator.coef_[1:][::-1], [estimator.intercept_]]


def apply_ransac_fit(arr: da.Array, time: da.Array, res: tuple[float, float], max_distance: float, degree: int):
    arr = da.array(arr)

    window_size = arr.shape[-1]
    window_mid = window_size // 2

    median_elev = da.median(arr, axis=0)
    mid_median_elev = (
        median_elev[:, :, window_mid, window_mid]
        .reshape(
            median_elev.shape[:2]
            + (
                1,
                1,
            )
        )
        .repeat(window_size, axis=2)
        .repeat(window_size, axis=3)
    )

    elev_mask = da.zeros_like(arr, dtype="bool")
    elev_mask[:, :, :, :, :] = (
        (da.abs(median_elev - mid_median_elev) < 1).reshape((1,) + median_elev.shape).repeat(arr.shape[0], axis=0)
    )

    distance_mask = da.zeros_like(arr, dtype="bool")

    distance_mask[:, :, :] = (
        (
            da.sqrt(
                da.sum(
                    da.dstack(
                        da.meshgrid(
                            ((da.arange(window_size) - window_mid) * res[0]) ** 2,
                            ((da.arange(window_size) - window_mid) * res[0]) ** 2,
                        )
                    ),
                    axis=2,
                )
            )
            < (max_distance / 2)
        )
        .reshape(
            (
                1,
                1,
                1,
            )
            + arr.shape[3:]
        )
        .repeat(arr.shape[0], axis=0)
        .repeat(arr.shape[1], axis=1)
        .repeat(arr.shape[2], axis=2)
    )
    # distance_mask = np.repeat(np.repeat((np.sqrt(np.sum(np.square(np.meshgrid(*([(np.arange(window_size) - window_mid) * res[0]] * 2))), axis=0)) < (max_distance / 2)).reshape((1, 1,) + median_elev.shape[2:]), median_elev.shape[0], axis=0), median_elev.shape[1], axis=1)

    # mask = (elev_mask & distance_mask).reshape((1,) + elev_mask.shape).repeat(arr.shape[0], axis=0)
    mask = elev_mask & distance_mask

    arr[mask] = np.nan

    heights = da.moveaxis(arr, 0, 2).reshape(median_elev.shape[:2] + (-1,))

    result = da.moveaxis(da.apply_along_axis(ransac_fit, axis=2, arr=heights, time=time, degree=degree), 2, 0)

    return result


def block_polyfit(rolled: xr.DataArray, res, max_distance, times, names):

    return (
        xr.apply_ufunc(
            apply_ransac_fit,
            rolled,
            input_core_dims=[
                (
                    "time",
                    "y",
                    "x",
                    "ry",
                    "rx",
                )
            ],
            output_core_dims=[("time", "y", "x")],
            exclude_dims=set(["rx", "ry", "time"]),
            dask="allowed",
            kwargs=dict(res=res, max_distance=max_distance, time=times, degree=len(names) - 1),
        )
        .rename(time="coef")
        .assign_coords(coef=names)
    )


def glacier_stack(glacier="tinkarp", force_redo: bool = False, verbose: bool = True, client: dask.distributed.Client | None = None):

    #if client is None:
    #    client = dask.distributed.Client()

    glacier_info = GLACIERS[glacier]

    temp_dir = get_temp_dir(glacier_info["region"])
    glacier_stack_dir = temp_dir.joinpath(f"glacier_stacks/{glacier}")
    glacier_stack_dir.mkdir(parents=True, exist_ok=True)

    nc_path = glacier_stack_dir.joinpath("stack.nc")
    poly_path = nc_path.with_stem(nc_path.stem + "_poly")
    stack_corr_path = poly_path.with_stem(poly_path.stem + "_corr")



    res = (5, 5)
    bounds = align_bounds(rio.coords.BoundingBox(*glacier_info["bounds"]), res=res)
    poi = shapely.geometry.box(*bounds)
    width = int((bounds.right - bounds.left) / res[0])
    height = int((bounds.top - bounds.bottom) / res[1])
    transform = rio.transform.from_bounds(*bounds, width, height)
    xr_coords = [
        ("y", np.linspace(bounds.bottom + res[1] / 2, bounds.top - res[1] / 2, height)[::-1]),
        ("x", np.linspace(bounds.left + res[0] / 2, bounds.right - res[0] / 2, width)),
    ]
    if not nc_path.is_file() or force_redo:

        dems_dir = temp_dir.joinpath("arcticdem_coreg")
        ref_dem_path, ref_dem_years_path = build_npi_mosaic(region=glacier_info["region"])
        stable_terrain_path = build_stable_terrain_mask(region=glacier_info["region"])

        if "rgi_id" in glacier_info:
            rgi = gpd.read_file("zip:///home/erik/Projects/UiO/HeerLandSurges/input/RGI7/RGI2000-v7.0-G-07_svalbard_jan_mayen.zip!RGI2000-v7.0-G-07_svalbard_jan_mayen.shp").query(f"rgi_id == '{glacier_info['rgi_id']}'").to_crs(32633).iloc[0]

            rgi = rasterio.features.rasterize([rgi.geometry], out_shape=(height, width), transform=transform, default_value=1)
        else:
            rgi = None

        i = 0
        arrays = []
        dem_paths = list(dems_dir.iterdir())
        # arrays = np.empty((len(dem_paths), height, width), dtype="float32") + np.nan
        arrays = []
        times = []

        for filepath in tqdm(list(dems_dir.iterdir()), desc="Sampling rasters"):
            if filepath.suffix != ".tif":
                continue

            # if i > 10:
            #    break

            with rio.open(filepath) as raster:
                if not poi.intersects(shapely.geometry.box(*raster.bounds)):
                    continue

                window = rio.windows.from_bounds(*poi.bounds, transform=raster.transform)
                data = raster.read(1, window=window, masked=True, boundless=True).filled(np.nan)

            if np.count_nonzero(np.isfinite(data)) == 0:
                continue

            date = pd.to_datetime(filepath.stem.split("_")[3])

            year_dec = date.year + (date.month / 12) + (date.day / (30 * 12)) + (i / (365 * 24))

            # arrays[i, :, :] = data
            arrays.append(data)
            times.append(year_dec)

            i += 1

        stack = xr.Dataset({"ad_elevation": xr.DataArray(arrays, coords=[("time", times)] + xr_coords)})
        stack = stack.sortby("time")

        for name, path in [("stable_terrain", stable_terrain_path), ("ref_dem", ref_dem_path), ("ref_dem_year", ref_dem_years_path)]:
            
            with rio.open(path) as raster:
                window = rio.windows.from_bounds(*poi.bounds, transform=raster.transform)
                data = raster.read(1, window=window, masked=True, boundless=True).filled(0)

            # stack[name] = xr.DataArray(data, coords=xr_coords)
            stack[name] = ("y", "x"), data

        if rgi is not None:
            stack["rgi_mask"] = ("y", "x"), rgi

        median_elevation = stack["ad_elevation"].median("time")

        diffs = (stack["ad_elevation"] - median_elevation).where(stack["stable_terrain"] == 1).median(["x", "y"])
        stack["ad_elevation"] = stack["ad_elevation"] - diffs

        stack.to_netcdf(
            nc_path, encoding={key: {"zlib": True, "complevel": 9} for key in stack.data_vars}, engine="h5netcdf"
        )
    else:
        stack = xr.open_dataset(nc_path, chunks={"x": 10, "y": 10})

    raise NotImplementedError()

    if glacier == "tinkarp":
        pois = {
            "terminus": {"x": 551090, "y": 8658163},
            "terminus_b": {"x": 551910, "y": 8657711},
            "accumulation": {"x": 550250.90, "y": 8657357.22},
            "off_glacier": {"x": 553000, "y": 8656000},
        }
    elif glacier == "dronbreen":
        pois = {
            "accumulation_prob": {"x": 539486, "y": 8668589},
        }

    else:
        raise NotImplementedError()

    width = 400
    poi_slices = {
        k: {"x": slice(v["x"] - width / 2, v["x"] + width / 2), "y": slice(v["y"] + width / 2, v["y"] - width / 2)}
        for k, v in pois.items()
    }
    degree = 2
    names = ["poly_" + ("abcdefghijk"[i] if i != (degree) else "bias") for i in range(degree + 1)]

    max_distance = 200

    # stack["ad_elevation_med_temp"] = stack["ad_elevation"].median("time")
    stack["t"] = stack["time"] - 2021
    #location = stack.sel(**poi_slices["accumulation_prob"])
    location = stack

    chunking = {"x": 10, "y": 10}

    window_size = int(np.ceil(max_distance / res[0]))
    if window_size % 2 == 0:
        window_size += 1

    warnings.simplefilter("error")

    times = da.array(location.t.values).repeat(window_size**2)

    rolled = (location["ad_elevation"].rolling(x=window_size, y=window_size, center=True).construct(y="ry", x="rx")).chunk(**chunking, rx=window_size, ry=window_size).to_dataset()

    template = xr.DataArray(
        np.zeros((len(names), location.y.shape[0], location.x.shape[0])),
        coords=[("coef", names), location.y, location.x],
    ).chunk(chunking)
    import zarr
    output_path = Path("./polyfit_test.nc")
    #output_path = Path("./polyfit_test.zarr")
    with tempfile.TemporaryDirectory() as temp_dir:
        rolled_path = Path(temp_dir).joinpath("rolled.zarr")

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        
        if verbose:
            print("Writing rolled to a temporary file")
        task = rolled.to_zarr(rolled_path, encoding={v: {"compressor": compressor} for v in rolled.data_vars}, compute=True)

        #with TqdmCallback(desc="Writing rolled to a temporary file", smoothing=0.1):
        #    task.compute()

        if verbose:
            print("Running polyfit")

        with xr.open_zarr(rolled_path)["ad_elevation"] as rolled:
            polyfit = xr.map_blocks(
                block_polyfit,
                rolled,
                kwargs=dict(times=times.compute(), res=res, max_distance=max_distance, names=names),
                template=template,
            ).to_dataset(name="polyfit")
            polyfit["coef"] = polyfit["coef"].astype(str)


            # if output_path.is_dir():
            #     shutil.rmtree(output_path)


            # with TqdmCallback():
            #     polyfit = polyfit.compute(scheduler="processes")

            # task = polyfit.chunk(x=256, y=256, coef=3).to_zarr(output_path, compute=False)

            # with TqdmCallback():
            #     task.compute()
            # return

            #TODO: There's a huge bottleneck here. scheduler="processes" is much much faster, but netcdf cannot be written concurrently
            # A much better alternative would be to write to a zarr, but that errors out for some weird reason.

            print("Processing")
            task = polyfit.to_netcdf(output_path, compute=False)# encoding={v: {"compressor": compressor} for v in polyfit.data_vars}, compute=False)
            with TqdmCallback():
                task.compute()


    polyfit = xr.open_dataset(output_path)

    polyfit["polyfit"].isel(coef=0).plot()
    plt.show()


def main(client):
    process_all(show_progress_bar=False)
    # glacier_stack()



if __name__ == "__main__":
    #client = dask.distributed.Client("127.0.0.1:8786")
    #glacier_stack("dronbreen", client=None)
    main(None)
