import shapely
import dotenv
import xarray as xr
import netCDF4
from pathlib import Path
import json
import tempfile
import rasterio as rio
import pyproj
import pandas as pd
import geopandas as gpd
import functools
from variete import VRaster

# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.dask import TqdmCallback
from pqdm.processes import pqdm
from bounded_pool_executor import BoundedProcessPoolExecutor
import concurrent.futures
import numpy as np

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS

REGIONS = {"heer": {"xmin_lat": 16.61716, "ymax_lat": 78.00398, "xmax_lat": 18.42673, "ymin_lat": 77.58121}}


def download_icesat2(region: dict[str, float], start_date: str = "2018-01-01", end_date: str = "2023-01-01"):

    # This import is slow and is only needed here
    import icepyx

    dotenv.load_dotenv(".env")
    response = icepyx.Query(
        product="ATL06",
        spatial_extent=[region[k] for k in ["xmin_lat", "ymin_lat", "xmax_lat", "ymax_lat"]],
        date_range=[start_date, end_date],
        version="005",
    )
    response.earthdata_login()

    data_dir = Path("data").joinpath("ICESat-2")

    raw_files_dir = data_dir.joinpath("download")

    response.download_granules(raw_files_dir)

    return raw_files_dir


def read_h5(filepath) -> xr.Dataset:

    reader = netCDF4.Dataset(filepath, diskless=True, persist=False)

    ground_tracks = []
    transformer = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), pyproj.CRS.from_epsg(32633))
    for key in reader.groups.keys():
        if not key.startswith("gt"):
            continue
        ground_track = reader.groups.get(key)
        if "land_ice_segments" not in ground_track.groups.keys():
            continue

        dataset = (
            xr.open_dataset(xr.backends.NetCDF4DataStore(ground_track.groups.get("land_ice_segments")))
            .rename({"delta_time": "time"})
            .reset_coords()
        )
            
        eastings, northings = transformer.transform(dataset.latitude.values,dataset.longitude.values)
        for name, arr in [("easting", eastings), ("northing", northings)]:
            arr[~np.isfinite(arr)] = np.nan
            dataset[name] = ("time", arr.ravel())
        ground_tracks.append(dataset.expand_dims({"track": [key]}))

    if len(ground_tracks) == 0:
        return xr.Dataset()

    ground_tracks = xr.merge(ground_tracks).dropna("time", how="all", subset=["h_li"])

    return ground_tracks


def merge_icesat():
    data_dir = Path("data").joinpath("ICESat-2")
    temp_dir = Path("temp")

    output_path = temp_dir.joinpath("merged_icesat.nc")

    raw_files_dir = data_dir.joinpath("download")

    filepaths = list(raw_files_dir.glob("*.h5"))
    data = xr.concat(
        [ds for ds in process_map(read_h5, filepaths, desc="Loading h5 files") if len(ds) > 0], dim="time"
    ).sortby("time")

    with TqdmCallback(desc=f"Saving {output_path.name}"):
        data.chunk(time=int(1e5)).to_netcdf(
            output_path, encoding={k: {"zlib": True, "complevel": 5} for k in data.data_vars}, compute=False
        ).compute()


def grid_icesat():

    import pdal

    temp_dir = Path("temp")
    input_path = temp_dir.joinpath("merged_icesat.nc")


    data = xr.open_dataset(input_path).stack(all=["track", "time"])
    data = data.where(data["atl06_quality_summary"] == 0.)

    test_point = {"x": 547307, "y": 8638991}
    test_point = {"x": 547422, "y": 8639341}

    thresh = 50
    data = data.where(((xr.apply_ufunc(np.abs, data["northing"] - test_point["y"])) < thresh) & ((xr.apply_ufunc(np.abs, data["easting"] - test_point["x"])) < thresh), drop=True)

    print(data["time"].values)
    import matplotlib.pyplot as plt

    plt.scatter(data["time"], data["h_li"])
    plt.show()

    #print(data.set_index({"time": ["northing", "easting"]}))

    return

    #print(data.isel(time=0).load())
    #return
    data = np.vstack((data["easting"].values, data["northing"].values, data["h_li"].values)).T

    data = data[np.all(np.isfinite(data), axis=1)]

    data = np.core.records.fromarrays(data.T,names=["X", "Y", "Z"]) 


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_tif = Path(temp_dir).joinpath("grid.tif") 
        temp_tif = "dem.tif"
        pipeline = [
            {
                "type": "writers.gdal",
                "resolution": 50,
                "filename": str(temp_tif),
                "data_type": "float32",
                "override_srs": "EPSG:32633",
                "gdalopts": ["COMPRESS=DEFLATE", "TILED=YES", "ZLEVEL=5"],
                
            }
        ]
        pipe = pdal.Pipeline(json.dumps(pipeline), arrays=[data])

        pipe.execute()
        #print(pipe.arrays)

        with rio.open(temp_tif) as raster:
            print(raster.descriptions)
            for i, name in enumerate(raster.descriptions, start=1):
                print(name, raster.read(i).max())


@adsvalbard.caching.cache_nc(cache_label="region_label")
def get_is2_data(region_label: str, bounds: rio.coords.BoundingBox, crs: rio.CRS | None = None):

    if crs is None:
        crs = rio.CRS.from_epsg(CONSTANTS.crs_epsg)


    parts = []
    for filename in ["snow_free_svalbard.csv", "snow_cover_svalbard.csv"]:
        data = pd.read_csv(adsvalbard.utilities.get_data_dir("IC2").joinpath(filename))
        data = gpd.GeoDataFrame(data, geometry=data["geometry"].apply(shapely.wkt.loads), crs=4326).to_crs(crs)
        data["easting"] = data.geometry.x
        data["northing"] = data.geometry.y
        data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m%d")

        data = data[
            (data["easting"] <= bounds.right) &
            (data["easting"] >= bounds.left) &
            (data["northing"] >= bounds.bottom) &
            (data["northing"] <= bounds.top)
        ]

        data["on_snow"] = int("free" not in filename) 

        parts.append(data)

    data = xr.Dataset.from_dataframe(pd.concat(parts).drop(columns=["geometry", "p_b"]))
    

    return data.chunk(index=512)

    with xr.set_options(display_max_rows=100):
        print(data)

    
    
@adsvalbard.caching.cache_feather(subdir="is2_data", cache_label="cache_label")
def filter_is2_data(bounds: rio.coords.BoundingBox, dem_data: pd.Series, is2_data: xr.Dataset, cache_label: str | None = None, keep_columns: list[str] = ["easting", "northing", "h_te_best_fit"]) -> pd.DataFrame:

    
    # The filtering stage will take forever if these are not in memory
    is2_data["easting"].load()
    is2_data["northing"].load()
    is2_subset = is2_data.where(
        (is2_data["easting"] <= bounds.right) &
        (is2_data["easting"] >= bounds.left) & 
        (is2_data["northing"] >= bounds.bottom) &
        (is2_data["northing"] <= bounds.top),
        drop=True, 
    )
    is2_subset["date"].load()
    is2_subset["on_snow"].load()
    close_in_time = np.abs(is2_subset["date"] - pd.to_datetime(dem_data["datetime"]).to_datetime64()) < pd.Timedelta(days=30 * 6) 
    is2_subset = is2_subset.where(close_in_time | (is2_subset["on_snow"] == 0), drop=True)

    return is2_subset[keep_columns].to_pandas().reset_index()
           


def icesat_main():
    region = REGIONS["heer"]

    #merge_icesat()
    grid_icesat()
