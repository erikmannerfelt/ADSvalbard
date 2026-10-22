import concurrent.futures
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import dask

import dask.dataframe as dd
import geopandas as gpd
import geoutils as gu
import numba
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import rasterio.warp
import requests
import scipy.ndimage
import scipy.optimize
import shapely.geometry
import skimage
from osgeo import gdal
from pyproj import CRS
from tqdm import tqdm

import adsvalbard.utilities
from adsvalbard.rasters import build_npi_mosaic, build_stable_terrain_mask

with warnings.catch_warnings():
    warnings.simplefilter("ignore", numba.NumbaDeprecationWarning)
    import xdem

warnings.simplefilter("ignore", RuntimeWarning)
np.seterr(all="ignore")

gdal.UseExceptions()
REGION = "svalbard"


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON encoder that properly serializes numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path("data/").absolute()


def get_data_urls():
    np_dem_base_url = (
        "https://public.data.npolar.no/kartdata/S0_Terrengmodell/Delmodell/"
    )

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


def download_file(url: str, output_path: Path):
    with requests.get(
        url, stream=True
    ) as request, tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("file")
        if request.status_code != 200:
            raise ValueError(f"{request.status_code=} {request.content=}")

        with open(temp_file, "wb") as outfile:
            shutil.copyfileobj(request.raw, outfile)

        shutil.move(temp_file, output_path)


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
        bounds = {
            key: getattr(bounds, key) for key in ["left", "bottom", "right", "top"]
        }

    if res is None:
        res = get_res()
    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):
            mod = (bounds[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds[bound] = (
                bounds[bound]
                - mod
                + (res[i] if i > 0 and mod != 0 else 0)
                + ((buffer or 0) * (1 if i > 0 else -1))
            )

    return rio.coords.BoundingBox(**bounds)


def get_crs() -> CRS:
    """Get the target Coordinate Reference System (CRS)."""
    return CRS.from_epsg(32633)


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
            "svalbard": {
                "left": 341002.5,
                "bottom": 8455002.5,
                "right": 905002.5,
                "top": 8982002.5,
            },
            "nordenskiold": {
                "left": 443002.5,
                "bottom": 8626007.5,
                "right": 560242.5,
                "top": 8703007.5,
            },
            "heerland": {
                "left": 537010,
                "bottom": 8602780,
                "right": 582940,
                "top": 8656400,
            },
            "kongsvegen": {
                "left": 438960,
                "bottom": 8735190,
                "right": 470420,
                "top": 8764230,
            },
            "haakonvii": {
                "left": 412970,
                "bottom": 8689380,
                "right": 495820,
                "top": 8837790,
            },
        }

        bounds = region_bounds[region]

        return align_bounds(bounds, res=res, half_mod=half_mod)

    crs = get_crs()
    roi_path = Path("shapes/region_of_interest.geojson")
    roi = gpd.read_file(roi_path).to_crs(crs)
    res = get_res()
    return adsvalbard.utilities.align_bounds(
        rasterio.coords.BoundingBox(*roi.total_bounds), res=res, half_mod=False
    )


def get_temp_dir(label: str = REGION) -> Path:
    """Get (and create) the path to the directory to store temporary files in."""

    temp_dir = get_data_dir().parent.joinpath(f"temp.{label}")
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


def get_strips() -> gpd.GeoDataFrame:
    temp_dir = get_temp_dir()

    output_path = temp_dir.joinpath("strip-meta.feather")

    if output_path.is_file():
        return gpd.read_feather(output_path)

    bounds = get_bounds()
    crs = get_crs()
    bounds_wgs = (
        gpd.GeoSeries(shapely.geometry.box(*list(bounds)), crs=get_crs())
        .to_crs(4326)
        .total_bounds
    )

    # meta_url, filepath = get_data_urls()["strip_metadata"][0])

    meta_url, filepath = [
        v for v in get_data_urls()["strip_metadata"] if "geocell_metadata" in str(v[1])
    ][0]

    if not filepath.is_file():
        filepath.parent.mkdir(exist_ok=True, parents=True)
        download_file(meta_url, filepath)

    with open(filepath) as infile:
        data_json = json.load(infile)

    data = pd.DataFrame.from_records(data_json["links"])
    data = data[data["title"].str.contains("Geocell")]
    data["title"] = data["title"].str.replace("Geocell ", "")

    data["lat"] = data["title"].str.slice(1, 3).astype(int)
    data["lon"] = data["title"].str.slice(4, 7).astype(int) * (
        (data["title"].str.slice(-4, -3) == "e") * 2 - 1
    )
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
            files = pd.DataFrame.from_records(json.load(infile)["links"])[
                ["title", "href"]
            ]
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
        out["dem"] = os.path.join(
            os.path.dirname(self_link["href"]), data["assets"]["dem"]["href"][2:]
        )
        out["matchtag"] = os.path.join(
            os.path.dirname(self_link["href"]), data["assets"]["matchtag"]["href"][2:]
        )

        return out

    strips = gpd.GeoDataFrame.from_records([load(fp) for fp in filepaths])
    strips = strips.set_geometry(
        strips["geometry"].apply(shapely.geometry.shape), crs=CRS.from_epsg(4326)
    )
    strips["geometry"] = gpd.GeoSeries.from_wkb(
        strips["geometry"].to_wkb(output_dimension=2)
    )
    strips = strips.to_crs(crs)

    for col in ["pgc:image_ids", "pgc:avg_sun_elevs", "instruments"]:
        strips[col] = strips[col].apply(lambda col_list: ",".join(map(str, col_list)))

    strips.to_feather(output_path)
    return strips


def now_time() -> str:
    """Return a string showing the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")


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
    return {
        "COMPRESS": "DEFLATE",
        "ZLEVEL": "12",
        "TILED": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }


def create_warped_vrt(
    filepath: Path | str,
    vrt_filepath: Path | str,
    out_crs: str | CRS,
    in_crs: str | CRS | None = None,
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

    vrt = gdal.AutoCreateWarpedVRT(
        ds, in_crs, out_crs, rasterio.warp.Resampling.bilinear
    )
    vrt.GetDriver().CreateCopy(str(vrt_filepath), vrt)

    del ds
    del vrt


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
    struct = scipy.ndimage.iterate_structure(
        scipy.ndimage.generate_binary_structure(2, 1), 3
    )
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
    outlier_mask = np.isin(
        labelled, unique[counts > int(counts[np.argmax(counts)] * 0.01)], invert=True
    )

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
                mask = raster.read_masks(
                    1,
                    out_shape=(
                        int(raster.height / overview_level),
                        int(raster.width / overview_level),
                    ),
                )

                # Generate approximate coordinates of the pixels in the wrong CRS
                coords = np.array(
                    np.meshgrid(
                        np.linspace(
                            raster.bounds.left, raster.bounds.right, mask.shape[1]
                        ),
                        np.linspace(
                            raster.bounds.bottom, raster.bounds.top, mask.shape[0]
                        )[::-1],
                    )
                )

                # Convert the coordinates to the right CRS and mask them to all data pixels only
                coords = gpd.points_from_xy(
                    *coords.reshape((2, -1)).T[mask.ravel() > 0].T, crs=raster.crs
                ).to_crs(crs)

                # Find the smallest "right CRS" bounds of the unmasked pixels in the raster
                warp_bounds = align_bounds(
                    rio.coords.BoundingBox(*coords.total_bounds), res=res, buffer=res[0]
                )

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
    stable_terrain_mask = gu.Raster(str(stable_terrain_path), load_data=False).crop(
        tba_dem, inplace=False
    )
    stable_terrain_mask = stable_terrain_mask.data.filled(0) == 1

    if np.count_nonzero(stable_terrain_mask) == 0:
        raise ValueError("No stable terrain pixels in window")
    if (
        np.count_nonzero(np.isfinite(tba_dem.data.filled(np.nan)[stable_terrain_mask]))
        == 0
    ):
        raise ValueError("No overlapping stable terrain")
    if verbose:
        print(f"\t{now_time()}: Loaded stable terrain mask")

    del mask

    npi_mosaic_path, _ = build_npi_mosaic(verbose=verbose)
    ref_dem = gu.Raster(str(npi_mosaic_path), load_data=False).crop(
        tba_dem, inplace=False
    )
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
                "name": re.search(r"coreg.*'", str(part.__class__))
                .group()
                .replace("'", "")
                .replace("coreg.", ""),
                "meta": part.__dict__,
            }
        )

    if verbose:
        print(f"\t{now_time()}: Finished co-registration. Transforming DEM")
    tba_dem = coreg.apply(tba_dem)

    diff = tba_dem.data[stable_terrain_mask].filled(np.nan) - ref_dem.data[
        stable_terrain_mask
    ].filled(np.nan)
    metadata["stable_nmad"] = float(xdem.spatialstats.nmad(diff))
    metadata["stable_bias"] = float(np.nanmedian(diff))
    metadata["stable_fraction"] = float(
        np.count_nonzero(stable_terrain_mask) / np.prod(stable_terrain_mask.shape)
    )
    metadata["pixel_count"] = int(np.count_nonzero(np.isfinite(tba_dem.data)))
    metadata["bounds"] = {
        b: float(getattr(tba_dem.bounds, b)) for b in ["left", "bottom", "right", "top"]
    }
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
    npi_dem_years = (
        xdem.DEM(str(npi_mosaic_years_path), load_data=False)
        .crop(dem, inplace=False)
        .astype("float32")
    )
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


def process_all(show_progress_bar: bool = True):
    strips = get_strips()

    # All of Nordenskiöld Land
    poi = shapely.geometry.box(*get_bounds(half_mod=False))

    # Build these first (or conversely, check that they exist)
    build_npi_mosaic(verbose=(not show_progress_bar))
    build_stable_terrain_mask(verbose=(not show_progress_bar))

    # Remove the northern part of Isfjorden for Nordenskiold because it's too small to use.
    if REGION == "nordenskiold":
        poi = poi.difference(shapely.geometry.box(435428, 8679301, 497701, 8714978))

    failures_file = Path(f"failures.{REGION}.csv")

    strips = strips[strips.intersects(poi)].sort_values(
        "start_datetime", ascending=False
    )
    if failures_file.is_file():
        failures = pd.read_csv(failures_file, names=["title", "exception"])
        strips = strips[~strips["title"].isin(failures["title"])]

    strips["start_datetime"] = pd.to_datetime(strips["start_datetime"])

    skip_paths = [
        "SETSM_s2s041_W1W1_20190719_1020010085AAE100_10200100863B9000_2m_lsf_seg1_dem",
        "SETSM_s2s041_W1W3_20180405_10200100706CF700_104001003CAF7A00_2m_lsf_seg1_dem",
        "SETSM_s2s041_WV01_20170522_10200100622AE000_1020010060E84300_2m_lsf_seg1",
        "SETSM_s2s041_WV03_20150316_10400100085B1D00_1040010009170400_2m_lsf_seg6",
    ]

    # Fix for a very strange bug where urls were jumbled up
    for col in ["dem", "matchtag"]:
        strips[col] = strips[col].str.replace(r".*/tps://", "https://", regex=True) 

    with tqdm(total=strips.shape[0], disable=(not show_progress_bar)) as progress_bar:
        for _, dem_data in strips.iterrows():
            start_time = time.time()
            if show_progress_bar:
                progress_bar.set_description(f"Working on {dem_data['title']}")

            dem_path, _ = download_arcticdem(dem_data, verbose=(not show_progress_bar))
            if any(s in str(dem_path) for s in skip_paths):
                print(f"Skipping {dem_path}")
                continue
            try:
                dem_coreg = coregister(
                    dem_path=dem_path, verbose=(not show_progress_bar)
                )
            except Exception as exception:
                exception_str = str(exception)
                if (
                    exception.__class__.__name__ == "AssertionError"
                    and len(exception_str) == 0
                ):
                    with open(failures_file, "a+") as outfile:
                        outfile.write(
                            dem_data["title"]
                            + ',"Empty AssertionError; probably xdem"\n'
                        )
                    progress_bar.update()
                    if not show_progress_bar:
                        print(
                            f"{now_time()}: {dem_data['title']} failed. Recorded failure"
                        )
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
                        "boolean index did not match indexed array along dimension 0",
                    ]
                ):
                    raise exception
                with open(failures_file, "a+") as outfile:
                    outfile.write(
                        dem_data["title"]
                        + ',"'
                        + str(exception).replace("\n", " ")
                        + '"\n'
                    )
                if not show_progress_bar:
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


def main(client):
    process_all(show_progress_bar=False)


if __name__ == "__main__":
    main(None)
