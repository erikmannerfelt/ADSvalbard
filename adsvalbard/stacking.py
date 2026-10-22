import datetime
import shutil
import warnings
from pathlib import Path
import threading
import functools

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
import tqdm.dask
import tqdm.contrib.concurrent

import rasterio
import rasterio.coords
import rasterio.transform
import rasterio.windows

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS


def process_chunk(filepath: Path, input_filepaths: dict[Path, threading.Lock], outlier_proba_dir: Path, bounds: rasterio.coords.BoundingBox, res: float, crs: object, outlier_threshold: float, v_clip: float, progress_bar: tqdm.tqdm | None = None, vertical_shifts: pd.DataFrame | None = None):
    shape = adsvalbard.utilities.shape_from_bounds_res(bounds=bounds, res=[res] * 2)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, res, res)

    dtypes = {"median": "float32", "count": "uint8", "nmad": "float32", "doy": "uint16"}
    output_paths = {kind: filepath.with_stem(filepath.stem + (f"_{kind}" if kind != "median" else "")) for kind in dtypes}
    temp_paths = {kind: fp.with_name(fp.name + ".tmp") for kind, fp in output_paths.items()}

    orig_params = dict(
                driver="GTiff",
                width=shape[1],
                height=shape[0],
                count=1,
                crs=crs,
                transform=transform,
                compress="deflate",
                tiled=True,
                zlevel=12,
    )
    def write_params(kind: str) -> dict[str, object]:
        params = orig_params.copy()

        params["dtype"] = dtypes[kind]

        if dtypes[kind] == "float32":
            params.update(
                {
                    "nodata": -9999,
                }
            )
        elif kind == "doy":
            params["nodata"] = 0
        return params

    data = []
    doys = []

    X = None
    Y = None
    for path, _ in input_filepaths.items():
        with warnings.catch_warnings():
            if path.name.endswith("epsg32633_5.0m.vrt") and not path.with_stem(path.stem.replace("_dem_", "_matchtag_")).is_file():
                print(f"Matchtag missing for {path.stem} in chunk {'/'.join(output_paths['median'].parts[-2:])}")
                continue
            if vertical_shifts is not None:
                if path.stem.split("_dem_")[0] not in vertical_shifts.index:
                    continue

            with rasterio.open(path) as raster:
                raster_win = rasterio.windows.from_bounds(
                    *bounds, raster.transform
                )

                data.append(
                    np.clip(
                        raster.read(
                            1, window=raster_win, masked=True, boundless=True
                        ).filled(np.nan),
                        -v_clip,
                        v_clip,
                    )
                )

                win_transform = rasterio.windows.transform(raster_win, raster.transform)

                if X is None:
                    # Build column and row indices
                    height, width = data[-1].shape
                    cols = np.arange(width)
                    rows = np.arange(height)
                    xs = win_transform.c + cols * win_transform.a + win_transform.b * rows[0]  # but b is usually 0
                    ys = win_transform.f + rows * win_transform.e                          # e is usually negative

                    # Create 2D grids
                    X, Y = np.meshgrid(xs, ys)

            if path.name.endswith("epsg32633_5.0m.vrt"):
                with rasterio.open(path.with_stem(path.stem.replace("_dem_", "_matchtag_"))) as raster2:
                    data[-1][raster2.read(1, window=raster_win, masked=True, boundless=True).filled(0) == 0] = np.nan 

            if vertical_shifts is not None:
                a, b, c, center_x, center_y = vertical_shifts.loc[path.stem.split("_dem_")[0], ["a", "b", "c", "center_x", "center_y"]].values
                corr = a * (X - center_x) + b * (Y - center_y) + c
                data[-1] += corr

            date_str = path.stem.split("_")[3]
            year = date_str[:4]
            doy = datetime.date(int(year), int(date_str[4:6]), int(date_str[6:8])).timetuple().tm_yday
            outlier_proba_path = outlier_proba_dir / f"{year}/{path.stem}_outlier_proba.tif"

            doys.append(
                np.zeros(shape, "uint16") + doy,
            )

            if outlier_threshold is not None:
                with rasterio.open(outlier_proba_path) as raster:

                    data[-1][raster.read(1, window=raster_win, masked=True, boundless=True).filled(255) > int(outlier_threshold * 255)] = np.nan

    if len(data) == 0:
        median = np.zeros(shape, dtype="float32") + np.nan
        nmad = median.copy()
        count = np.zeros(shape=shape, dtype="uint8")
        doy = count.copy().astype("uint16")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice")
            finites = np.isfinite(data)
            median = np.nanmedian(data, axis=0)

            doy = np.ma.median(np.ma.masked_array(doys, mask=~finites), axis=0).filled(0).astype("uint16")
            count = np.count_nonzero(finites, axis=0)
            doy[count == 0] = 0
            nmad = 1.4826 * np.nanmedian(np.abs(data - median[None, :, :]), axis=0)

    del data

    for kind, arr in [("median", median), ("nmad", nmad), ("count", count), ("doy", doy)]:
        if dtypes[kind] == "float32":
            arr = np.where(np.isfinite(arr), arr, -9999)

        with rasterio.open(temp_paths[kind], "w", **write_params(kind=kind)) as raster:
            raster.write(arr, 1)

        shutil.move(temp_paths[kind], output_paths[kind])

    if progress_bar is not None:
        progress_bar.update()

def process_chunk_wrapper(args):
    return process_chunk(**args)


def get_chunk_filepath(i: int,output_vrt_path: Path, region: str = "svalbard", chunksize: int = 512 * 7):
    import rasterio as rio
    res = CONSTANTS.res
    bounds_dict = CONSTANTS.regions[region]
    bounds = rio.coords.BoundingBox(**bounds_dict)
    shape = adsvalbard.utilities.shape_from_bounds_res(bounds, [res] * 2)
    n_col_chunks = int(np.ceil(shape[1] / chunksize))
    n_row_chunks = int(np.ceil(shape[0] / chunksize))
    n_zero_pads = int(np.ceil(np.log10(max(n_col_chunks, n_row_chunks))) + 1)
    col = i % n_col_chunks
    row = int((i - col) / n_col_chunks) 

    return output_vrt_path.parent / f"{output_vrt_path.stem}_chunks_{chunksize}/chunk_{str(row).zfill(n_zero_pads)}_{str(col).zfill(n_zero_pads)}.tif"

def create_median_stack(
    years: int | list[int] | None = 2021,
    n_threads: int | None = None,
    region: str = "svalbard",
    raster_type: str = "dem",
    outlier_threshold: float | None = 0.95,
    verbose: bool = True,
    write_in_mem: bool = False,
    bounds_override: rasterio.coords.BoundingBox | None = None,
    vertcoreg_label: str | None = None,
    nchunks_override: int | None = None,
):
    import rasterio as rio
    import rasterio.transform
    import shapely.geometry
    import contextlib
    import random
    import concurrent.futures
    from osgeo import gdal

    import adsvalbard.arcticdem
    import adsvalbard.utilities
    import adsvalbard.rasters
    temp_dir = CONSTANTS.temp_dir.with_stem("temp.svalbard")

    if raster_type == "dem":
        raster_dir = temp_dir.joinpath("arcticdem_coreg")
    elif raster_type in ["dem_noncoreg", "dem_vertcoreg"]:
        raster_dir = temp_dir / "arcticdem_vrts"

    else:
        raise NotImplementedError(f"Unknown raster type: {raster_type}")

    if not 0. < outlier_threshold < 1.:
        raise ValueError(f"Outlier threshold needs to be between 0 and 1: given: {outlier_threshold}")

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

    filt_str = f"filt_{str(int(outlier_threshold * 100)).zfill(3)}"
    bad_titles = (CONSTANTS.exclusions_dir / "bad_dems.txt").read_text().splitlines()

    vertical_shifts = None

    if raster_type == "dem":
        output_path = temp_dir / f"medians/{region}/dem/median_{filt_str}_dem{ext}.vrt"
        v_clip = 1500
        pattern = "*_dem_coreg.tif"
        dirs = [raster_dir]
        bad_titles = list(set(bad_titles + (CONSTANTS.exclusions_dir / "bad_dems_dem.txt").read_text().splitlines()))
    elif raster_type == "dem_noncoreg":
        output_path = temp_dir / f"medians/{region}/dem_noncoreg/median_dem{ext}.vrt"
        v_clip = 1500
        pattern = "*_dem_epsg32633_5.0m.vrt"
        outlier_threshold = None
        dirs = [raster_dir]
    elif raster_type == "dem_vertcoreg":
        output_path = temp_dir / f"medians/{region}/dem_vertcoreg/median_dem{ext}.vrt"
        v_clip = 1500
        pattern = "*_dem_epsg32633_5.0m.vrt"
        outlier_threshold = None
        dirs = [raster_dir]

        if len(years) > 1:
            raise NotImplementedError("Vertical coreg not implemented for multiple years")

        if vertcoreg_label is not None:
            vertcoreg_label = f"_{vertcoreg_label}"
        else:
            vertcoreg_label = ""
        vertical_shifts = pd.read_csv(temp_dir / f"vertcoreg_results/vertcoreg_results_{years[0]}{vertcoreg_label}.csv", index_col=0).squeeze()
        vertical_shifts = vertical_shifts[vertical_shifts["n_comparisons"] > 60]

        bad_titles = list(set(bad_titles + (CONSTANTS.exclusions_dir / "bad_dems_dem_vertcoreg.txt").read_text().splitlines()))

    else:
        raise NotImplementedError(f"Unknown raster type: {raster_type}")

    strips = adsvalbard.arcticdem.get_strips("svalbard")
    strips = strips[pd.to_datetime(strips["datetime"]).dt.year.isin(years)]

    strips = strips[~strips["title"].isin(bad_titles)]

    res = CONSTANTS.res
    bounds_dict = CONSTANTS.regions[region]
    bounds = rio.coords.BoundingBox(**bounds_dict)
    shape = adsvalbard.utilities.shape_from_bounds_res(bounds, [res] * 2)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, res, res)

    crs = rio.CRS.from_epsg(CONSTANTS.crs_epsg)

    block_size = [512 * 7] * 2

    n_col_chunks = int(np.ceil(shape[1] / block_size[0]))
    n_row_chunks = int(np.ceil(shape[0] / block_size[0]))
    n_zero_pads = int(np.ceil(np.log10(max(n_col_chunks, n_row_chunks))) + 1)

    chunks = []
    for i, chunk_bounds in enumerate(adsvalbard.rasters.generate_raster_chunks(bounds=bounds, res=res, chunksize=block_size[0])):

        if bounds_override is not None:
            if not adsvalbard.utilities.bounds_intersect(bounds_override, chunk_bounds):
                continue

        col = i % n_col_chunks
        row = int((i - col) / n_col_chunks) 

        chunk_nr = f"chunk_{str(row).zfill(n_zero_pads)}_{str(col).zfill(n_zero_pads)}"

        # These chunks don't play well with the intersection filter
        if chunk_nr not in ["chunk_008_021", "chunk_018_021", "chunk_000_027", "chunk_001_028"]:
            # Filter out completely empty chunks
            if not strips.intersects(shapely.geometry.box(*chunk_bounds)).any():
                continue

        shape = adsvalbard.utilities.shape_from_bounds_res(chunk_bounds, [res] * 2)

        chunks.append(
            {
                "filepath": output_path.parent / f"{output_path.stem}_chunks_{block_size[0]}/{chunk_nr}.tif",
                "bounds": chunk_bounds,
                "input_filepaths": {},
            }
        )

    centerpoint = 549457, 8641637 # Kjellstromdalen
    centerpoint = 654172, 8883410 # Austfonna NW margin
    centerpoint = 673467, 8881915 # Austfonna camp
    def sort_chunk(chunk):

        easting = np.mean([chunk["bounds"].right, chunk["bounds"].left])
        northing = np.mean([chunk["bounds"].top, chunk["bounds"].bottom])

        return ((easting - centerpoint[0]) ** 2 + (northing - centerpoint[1]) ** 2) ** 2

    if nchunks_override is not None:
        chunks.sort(key=sort_chunk)
        chunks = chunks[:nchunks_override]

    chunks_to_process = [chunk for chunk in chunks if not chunk["filepath"].is_file()]
    chunks_to_process = []
    for chunk in sorted(chunks, key=sort_chunk):
        if chunk["filepath"].is_file():
            # If a tmp file is left, redo it. If there is none, then it's good
            if len(list(chunk["filepath"].parent.glob(f"{chunk['filepath'].stem}*.tmp"))) == 0:
                continue

        chunks_to_process.append(chunk)

    if len(chunks_to_process) > 0:
        chunks_to_process[-1]["filepath"].parent.mkdir(exist_ok=True, parents=True)

    locks = {}
    for chunk in chunks_to_process:
        overlapping_strips: gpd.GeoDataFrame = strips[strips.intersects(shapely.geometry.box(*chunk["bounds"]))]

        for _, strip in overlapping_strips.iterrows():
            if strip["title"] in bad_titles:
                print(f"Skipping {strip['title']}")
                continue
            for raster_dir in dirs:
                filepath = raster_dir / f"{strip.title}{pattern.replace('*', '')}"

                if not filepath.is_file():
                    continue

                if filepath not in locks:
                    locks[filepath] = threading.Lock()

                chunk["input_filepaths"][filepath] = None
                # chunk["input_filepaths"].append(c

    call_args = []
    for chunk in chunks_to_process:
        call_args.append({
            **chunk,
            "res": res,
            "crs": crs,
            "outlier_threshold": outlier_threshold,
            "v_clip": v_clip,
            "outlier_proba_dir": temp_dir / "outlier_proba",
            "vertical_shifts": vertical_shifts.copy() if vertical_shifts is not None else None,
        })

    if n_threads == 1 or len(call_args) == 1:
        with tqdm.tqdm(total=len(call_args)) as progress_bar:
            for args in call_args:
                process_chunk_wrapper(args)
                progress_bar.update(1)
    elif len(call_args) > 0:
        tqdm.contrib.concurrent.process_map(process_chunk_wrapper, call_args, max_workers=n_threads, smoothing=0.1)

    kinds = ["median", "nmad", "count", "doy"]

    gdal.UseExceptions()
    for kind in kinds:
        out_path = output_path.with_stem(output_path.stem + (f"_{kind}" if kind != "median" else ""))

        chunk_paths = []
        for chunk in chunks:
            chunk_path = chunk["filepath"].with_stem(chunk["filepath"].stem + (f"_{kind}" if kind != "median" else "")).absolute()
            if not chunk_path.is_file():
                continue
            chunk_paths.append(str(chunk_path))

        gdal.BuildVRT(str(out_path), chunk_paths)

    return [chunk["filepath"] for chunk in chunks]


def make_chunk_polygons():
    import rasterio as rio
    import adsvalbard.rasters
    import shapely.geometry

    temp_dir = CONSTANTS.temp_dir.with_stem("temp.svalbard")
    res = CONSTANTS.res
    bounds = rio.coords.BoundingBox(**CONSTANTS.regions["svalbard"])
    chunksize = 512 * 7

    # Exclude Kong Karls Land and Hopen
    exclusion_box = shapely.geometry.box(744600, 8455000, 904000,8811000)
    chunks = []
    for i, chunk_bounds in enumerate(adsvalbard.rasters.generate_raster_chunks(bounds=bounds, res=res, chunksize=chunksize)):
        chunk_id = get_chunk_filepath(i, Path("/")).stem

        if exclusion_box.intersects(shapely.geometry.box(*chunk_bounds)):
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "geometry": shapely.geometry.box(*chunk_bounds),
            }
        )

    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(chunks), geometry="geometry", crs=CONSTANTS.crs_epsg)

    gdf.to_file(temp_dir / "chunk_outlines.geojson")
    return gdf


def remove_bad_dem_chunks():
    import os
    import rasterio as rio
    import shapely.geometry
    import adsvalbard.arcticdem
    import adsvalbard.rasters

    strips = adsvalbard.arcticdem.get_strips("svalbard")
    temp_dir = CONSTANTS.temp_dir.with_stem("temp.svalbard")
    bad_titles = (CONSTANTS.exclusions_dir / "bad_dems.txt").read_text().splitlines()
    bad_coreg_titles = (CONSTANTS.exclusions_dir / "bad_dems_dem.txt").read_text().splitlines()
    bad_vertcoreg_titles = (CONSTANTS.exclusions_dir / "bad_dems_dem_vertcoreg.txt").read_text().splitlines()

    bad_titles = {
        "dem_noncoreg": bad_titles,
        "dem_vertcoreg": list(set(bad_titles + bad_vertcoreg_titles)),
        "dem": bad_coreg_titles,
    }

    # Override the above (2026-03-17)
    bad_titles["dem_vertcoreg"] = bad_vertcoreg_titles

    strips["datetime"] = pd.to_datetime(strips["datetime"])
    strips["year"] = strips["datetime"].dt.year

    res = CONSTANTS.res
    bounds = rio.coords.BoundingBox(**CONSTANTS.regions["svalbard"])

    chunksize = 512 * 7

    filepaths_to_remove = []
    print("Looking for paths")
    for year in range(2013, 2025):
        # Extract every year's strip
        year_strips = strips[strips["year"] == year]
        # Loop over all product types
        for product in ["dem", "dem_noncoreg", "dem_vertcoreg"]:
            # Extract only the bad strips for this year and product
            bad_strips = year_strips[year_strips["title"].isin(bad_titles[product])]
            # Skip if there are no bad strips
            if bad_strips.shape[0] == 0:
                continue

            # Loop through all chunks
            for i, chunk_bounds in enumerate(adsvalbard.rasters.generate_raster_chunks(bounds=bounds, res=res, chunksize=chunksize)):
                # Extract only the strips that may be involved in the chunk
                overlapping_strips = bad_strips[bad_strips.intersects(shapely.geometry.box(*chunk_bounds))]
                # Skip if there are no bad and overlapping strips
                if overlapping_strips.shape[0] == 0:
                    continue

                # Plan to remove both the filtered versions
                thresholds = ["_filt_075", "_filt_095"] if product == "dem" else [""]
                for threshold in thresholds:

                    # This will be something like path/to/chunk_000_002.tif
                    base_filepath = get_chunk_filepath(i, temp_dir / f"medians/svalbard/{product}/median{threshold}_dem_{year}")

                    # Try to find all e.g. path/to/chunk_000_002* files (e.g. _nmad.tif, _count.tif etc)
                    for subpath in base_filepath.parent.glob(f"{base_filepath.stem}*"):
                        filepaths_to_remove.append(subpath)

    n_to_remove = len(filepaths_to_remove)

    if n_to_remove == 0:
        print("No paths to remove")
        return

    yesno = input(f"Remove {n_to_remove} filepaths? y/[N]")

    if yesno.strip() != "y":
        print("Stopping")
        return

    for filepath in tqdm.tqdm(filepaths_to_remove, desc="Removing"):
        os.remove(filepath)


