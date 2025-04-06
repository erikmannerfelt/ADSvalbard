import itertools
import pickle
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.coords
import rasterio.features
import rasterio.transform
import rasterio.warp
import rasterio.windows
import shapely.geometry
from osgeo import gdal
from tqdm import tqdm

import adsvalbard.inputs
import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS
from adsvalbard.utilities import now_time

gdal.UseExceptions()


def vrt_warp(
    output_filepath: Path | str,
    input_filepath: Path | str,
    # src_crs: CRS | int | str | None = None,
    dst_crs: rio.CRS | int | str | None = None,
    dst_res: tuple[float, float] | float | None = None,
    # src_res: tuple[float, float] | None = None,
    dst_shape: tuple[int, int] | None = None,
    # src_bounds: BoundingBox | list[float] | None = None,
    dst_bounds: rasterio.coords.BoundingBox | list[float] | None = None,
    # src_transform: Affine | None = None,
    dst_transform: rio.Affine | None = None,
    src_nodata: int | float | None = None,
    dst_nodata: int | float | None = None,
    resampling: rasterio.warp.Resampling | str = "bilinear",
    multithread: bool = False,
) -> None:
    if isinstance(resampling, str):
        resampling = getattr(rasterio.warp.Resampling, resampling)

    kwargs = {
        "resampleAlg": adsvalbard.utilities.resampling_rio_to_gdal(resampling),  # type: ignore
        "multithread": multithread,
        "format": "VRT",
        "dstNodata": dst_nodata,
        "srcNodata": src_nodata,
    }

    # This is strange. Warped pixels that are outside the range of the original raster get assigned to 0
    # Unclear if this can be overridden somehow! It should be dst_nodata or np.nan
    if kwargs["dstNodata"] is None:
        kwargs["dstNodata"] = 0

    for key, crs in [("dstSRS", dst_crs)]:
        if crs is None:
            if key == "dst_wkt":
                raise TypeError("dst_crs has to be provided")
            continue
        if isinstance(crs, int):
            kwargs[key] = rio.CRS.from_epsg(crs).to_wkt()
        elif isinstance(crs, rio.CRS):
            kwargs[key] = crs.to_wkt() # type: ignore
        else:
            kwargs[key] = crs

    if dst_transform is not None and dst_shape is None:
        raise ValueError("dst_transform requires dst_shape, which was not supplied.")
    if dst_transform is not None and dst_res is not None:
        raise ValueError("dst_transform and dst_res cannot be used at the same time.")
    if dst_transform is not None and dst_bounds is not None:
        raise ValueError("dst_transform and dst_bounds cannot be used at the same time.")

    if dst_shape is not None and dst_res is not None:
        raise ValueError("dst_shape and dst_res cannot be used at the same time.")

    if dst_transform is not None:
        # kwargs["dstTransform"] = dst_transform.to_gdal()
        kwargs["outputBounds"] = list(rasterio.transform.array_bounds(*dst_shape, dst_transform))  # type: ignore
    elif dst_bounds is not None:
        kwargs["outputBounds"] = list(dst_bounds)

    if dst_shape is not None:
        kwargs["width"] = dst_shape[1]
        kwargs["height"] = dst_shape[0]

    if dst_res is not None:
        if isinstance(dst_res, Sequence):
            kwargs["xRes"] = dst_res[0]  # type: ignore
            kwargs["yRes"] = dst_res[1]  # type: ignore
        else:
            kwargs["xRes"] = dst_res
            kwargs["yRes"] = dst_res

    gdal.Warp(str(output_filepath), str(input_filepath), **kwargs)



def vrt_mosaic(output_path: Path, input_paths: list[Path]) -> None:
    """
    Build a mosaic from the given input paths.

    Parameters
    ----------
    output_path
        The output path of the VRT mosaic.
    input_paths
        A list of paths to build the VRT mosaic from.
    """
    output_path.parent.mkdir(exist_ok=True)
    gdal.BuildVRT(str(output_path), [str(p.absolute()) for p in input_paths])

def build_npi_mosaic_chunk(filepath: Path, bounds: rasterio.coords.BoundingBox, dem_parts: list[tuple[rasterio.coords.BoundingBox, Path]]) -> tuple[Path, Path] | None:
    """
    Build one chunk of the NPI DEM and DEM year mosaic.

    See `build_npi_mosaic()` for the main function.

    Parameters
    ----------
    filepath
        The output filepath of the mosaicked DEM. The year raster path will be generated from this path.
    bounds
        The bounding box to generate the DEM/year raster in.
    dem_parts
        The parts to mosaic and their bounding boxes.

    Returns
    -------
    If at least one DEM overlaps with the bounds:
        Both the DEM mosaic and a raster of the years.
    Otherwise:
        None
    """
    years_filepath = filepath.with_stem(filepath.stem + "_years")

    if filepath.is_file() and years_filepath.is_file():
        return filepath, years_filepath

    out_shape = adsvalbard.utilities.get_shape(bounds)
    out_transform = adsvalbard.utilities.get_transform(bounds)

    dem = np.zeros(out_shape, dtype="float32") - 9999.
    years = np.zeros(out_shape, dtype="uint16")

    for dem_part in dem_parts:

        year = int(dem_part[1].stem.split("_")[3])

        if not adsvalbard.utilities.bounds_intersect(dem_part[0], bounds):
            continue

        with rio.open(dem_part[1]) as raster:

            window = rasterio.windows.from_bounds(*bounds, transform=raster.transform)

            arr = raster.read(1, masked=True, window=window, boundless=True)

            dem[~arr.mask] = arr.data[~arr.mask]
            years[~arr.mask] = year

    if np.all(dem == -9999.):
        return None

    filepath.parent.mkdir(exist_ok=True, parents=True)
    for path, arr in [(filepath, dem), (years_filepath, years)]:

        nodata = -9999. if "float" in str(arr.dtype) else 0

        with rio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, crs=rio.CRS.from_epsg(CONSTANTS.crs_epsg), transform=out_transform, dtype=arr.dtype, nodata=nodata, compress="deflate", tiled=True) as raster:
            raster.write(arr, 1)
        

    return filepath, years_filepath


def generate_raster_chunks(bounds: rasterio.coords.BoundingBox, res: float | None = None, chunksize: int = 10000) -> list[rasterio.coords.BoundingBox]:
    """
    Generate chunks of a bounding box that can be processed independently.

    Parameters
    ----------
    bounds
        The bounding box to split up in chunks
    res
        The resolution of the raster to process
    chunksize
        The (maximum) size in Y/X of the chunks

    Examples
    --------
    >>> bounds = rasterio.coords.BoundingBox(0., 0., 500., 1000.)
    >>> res = 10.
    >>> generate_raster_chunks(bounds, res, chunksize=50)
    [BoundingBox(left=0.0, bottom=500.0, right=500.0, top=1000.0), BoundingBox(left=0.0, bottom=0.0, right=500.0, top=500.0)]
    >>> generate_raster_chunks(bounds, res, chunksize=100)
    [BoundingBox(left=0.0, bottom=0.0, right=500.0, top=1000.0)]

    Returns
    -------
    A list of chunks with a maximum size of (chunksize, chunksize) pixels.
    """

    if res is None:
        res = CONSTANTS.res

    shape = adsvalbard.utilities.shape_from_bounds_res(bounds, [res] * 2)
    transform = adsvalbard.utilities.get_transform(bounds, (res, res))

    n_row_chunks = int(np.ceil(shape[0] / chunksize))
    n_col_chunks = int(np.ceil(shape[1] / chunksize))

    chunks: list[rasterio.coords.BoundingBox] = []
    for i in range(n_row_chunks):
        for j in range(n_col_chunks):
            start_col = j * chunksize
            end_col = min(start_col + chunksize, shape[1])

            width = end_col - start_col

            start_row = i * chunksize
            end_row = min(start_row + chunksize, shape[0])
            height = end_row - start_row

            if height == 0 or width == 0:
                raise ValueError()

            chunk_bounds = rasterio.coords.BoundingBox(*rasterio.windows.bounds(
                window=rasterio.windows.Window(start_col, start_row, width, height),  # type: ignore
                transform=transform, 
            ))

            chunks.append(chunk_bounds)

    return chunks
    


def build_npi_mosaic(roi_path: Path = Path("shapes/region_of_interest.geojson"), verbose: bool = True) -> tuple[Path, Path]:
    """
    Build DEM and DEM year mosaics of tiles downloaded from the NPI.

    Parameters
    ----------
    roi_path
        The path to the region of interest shape. The total bounds are used to derive the bounds of the mask.
        
    verbose
        Whether to print updates to the console

    Returns
    -------
    Paths to the NPI DEM and DEM year mosaic.
    """
    crs = rio.CRS.from_epsg(CONSTANTS.crs_epsg)
    roi = gpd.read_file(roi_path).to_crs(crs)

    dem_path = CONSTANTS.temp_dir / "npi_mosaic.vrt"
    years_path = CONSTANTS.temp_dir / "npi_mosaic_years.vrt"

    if dem_path.is_file() and years_path.is_file():
        return dem_path, years_path

    total_bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*roi.total_bounds), res=[CONSTANTS.res] * 2, half_mod=False)

    chunks = generate_raster_chunks(bounds=total_bounds)

    chunk_dir: Path = CONSTANTS.cache_dir / "npi_dems/chunks/"

    dem_paths = adsvalbard.inputs.download_category("NP_DEMs")

    vrt_dir: Path = CONSTANTS.cache_dir / "npi_dems/vrts/"
    vrt_dir.mkdir(exist_ok=True, parents=True)
    vrts: list[tuple[rasterio.coords.BoundingBox, Path]] = []
    for filepath in tqdm(dem_paths, desc="Generating VRTs", disable=(not verbose)):

        output_filepath = vrt_dir / f"{filepath.stem}.vrt"
        uri = f"/vsizip/{filepath}/{filepath.stem}/{filepath.stem.replace('NP_', '')}.tif"

        with rio.open(uri) as raster:
            dem_bounds = adsvalbard.utilities.align_bounds(
                bounds=adsvalbard.utilities.warp_bounds(raster.bounds, raster.crs, crs),
                res=[CONSTANTS.res] * 2,
                half_mod=False
            )
        vrt_warp(output_filepath, input_filepath=uri, dst_crs=crs, dst_res=CONSTANTS.res, dst_bounds=dem_bounds, multithread=True)

        vrts.append((dem_bounds, output_filepath))

    dem_chunk_paths = []
    year_chunk_paths = []

    for i, chunk in tqdm(list(enumerate(chunks)), desc="Generating DEM chunks"):
        filepath = chunk_dir / f"npi_mosaic_{str(i).zfill(3)}.tif"
        result = build_npi_mosaic_chunk(
            filepath=filepath,
            bounds=chunk, 
            dem_parts=vrts
        )
        if result is None:
            continue

        dem_chunk_paths.append(result[0])
        year_chunk_paths.append(result[1])

    vrt_mosaic(dem_path, dem_chunk_paths)
    vrt_mosaic(years_path, year_chunk_paths)

    return dem_path, years_path


def build_stable_terrain_mask(roi_path: Path = Path("shapes/region_of_interest.geojson"), verbose: bool = True) -> Path:
    """
    Build a stable terrain mask for the region of interest.

    The mask excludes areas within:

    - 1936 glacier outlines from Geyman et al., 2022
    - Water (rivers/lakes) according to the NPI S100 map
    - Moraines according to the NPI S100 map

    ... and includes all other areas on land according to the NPI S100 map

    Parameters
    ----------
    roi_path
        The path to the region of interest shape. The total bounds are used to derive the bounds of the mask.
    verbose: 
        Whether to print updates to the console

    Returns
    -------
    A path to the stable terrain mask.
    """
    out_filepath = CONSTANTS.temp_dir / "stable_terrain.tif"
    if out_filepath.is_file():
        return out_filepath

    if verbose:
        print(f"{now_time()}: Building stable terrain mask")

    crs = rio.CRS.from_epsg(CONSTANTS.crs_epsg)
    
    roi = gpd.read_file(roi_path).to_crs(crs)
    total_bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*roi.total_bounds), res=[CONSTANTS.res] * 2, half_mod=False)

    transform = adsvalbard.utilities.get_transform(total_bounds)
    shape = adsvalbard.utilities.shape_from_bounds_res(total_bounds, [CONSTANTS.res] * 2)

    outline_paths = adsvalbard.inputs.download_category("outlines")

    geyman_path = [p for p in outline_paths if "GAO_SfM" in str(p)][0]
    s100_path = [p for p in outline_paths if "NP_S100" in str(p)][0]

    geyman_1936 = (
        gpd.read_file(f"zip://{geyman_path}!GAO_SfM_1936_1938_v3.shp")
        .buffer(0)
        .to_frame()
        .dissolve()
        .to_crs(crs)
    )
    water = gpd.read_file(f"zip://{s100_path}!NP_S100_SHP/S100_Vann_f.shp").dissolve().to_crs(crs)
    moraines = gpd.read_file(f"zip://{s100_path}!NP_S100_SHP/S100_Morener_f.shp").dissolve().to_crs(crs)
    land = gpd.read_file(f"zip://{s100_path}!NP_S100_SHP/S100_Land_f.shp").dissolve().to_crs(crs)

    stable = land.difference(geyman_1936.union(water).union(moraines))
    del geyman_1936
    del water
    del moraines
    del land
    if verbose:
        print(f"{now_time()}: Loaded stable terrain features in memory")
    rasterized = rasterio.features.rasterize(stable.geometry, out_shape=shape, transform=transform, default_value=1)

    if verbose:
        print(f"{now_time()}: Rasterized features.")
    
    out_filepath.parent.mkdir(exist_ok=True, parents=True)
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
    
