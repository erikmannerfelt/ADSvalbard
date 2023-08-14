import adsvalbard.utilities
import adsvalbard.arcticdem
import adsvalbard.icesat2
import shapely.geometry
from adsvalbard.constants import CONSTANTS
import pandas as pd
import xarray as xr
from tqdm import tqdm
from variete import VRaster
from pathlib import Path
import xdem
import numpy as np

FAILURE_FILE = CONSTANTS.temp_dir.joinpath("failures.csv")

def register_failure(dem_id: str, exception: str):
    FAILURE_FILE.parent.mkdir(exist_ok=True)
    with open(FAILURE_FILE, "a+") as outfile:
        outfile.write(f'"{dem_id}","{exception}"\n')

def get_previous_failures() -> pd.DataFrame:
    if FAILURE_FILE.is_file():
        return pd.read_csv(FAILURE_FILE, names=["title", "exception"])
    return pd.DataFrame(columns=["title", "exception"])


def coregister_is2(dem: VRaster, dem_data: pd.Series, is2_data: xr.Dataset):


    is2_subset = is2_data.where(
        (is2_data["easting"] <= dem.bounds.right) &
        (is2_data["easting"] >= dem.bounds.left) & 
        (is2_data["northing"] >= dem.bounds.bottom) &
        (is2_data["northing"] <= dem.bounds.top),
        drop=True, 
    )
    is2_subset["date"].load()
    is2_subset["on_snow"].load()
    close_in_time = np.abs(is2_subset["date"] - pd.to_datetime(dem_data["datetime"]).to_datetime64()) < pd.Timedelta(days=30 * 6) 
    is2_subset = is2_subset.where(close_in_time | (is2_subset["on_snow"] == 0), drop=True)

    is2 = is2_subset[["easting", "northing", "h_te_best_fit"]].to_pandas()

    print(is2)
    


    raise NotImplementedError()


def process_strip(strip: pd.Series, is2_data: xr.Dataset, progress_bar: tqdm | None = None):


    if progress_bar is not None:
        progress_bar.set_description("Downloading DEM")
    dem_path, mask_path = adsvalbard.arcticdem.download_arcticdem(strip)


    dem = adsvalbard.arcticdem.get_warped_masked_vrt(dem_path=dem_path, mask_path=mask_path, res=CONSTANTS.coreg_res)

    if progress_bar is not None:
        progress_bar.set_description("Co-registering to IS2")

    # The filtering stage will take forever if these are not in memory
    is2_data["easting"].load()
    is2_data["northing"].load()
    coregister_is2(dem, dem_data=strip, is2_data=is2_data)


  
def process(region: str = "nordenskiold", progress_bar: bool = True):

    bounds = adsvalbard.utilities.get_bounds(region=region)
    strips = adsvalbard.arcticdem.get_strips(cache_label=region)

    poi = shapely.geometry.box(*list(bounds))
    strips.sort_values("start_datetime", ascending=False, inplace=True)

    failures = get_previous_failures()

    strips = strips[~strips["title"].isin(failures["title"])]

    is2_data = adsvalbard.icesat2.get_is2_data(cache_label=region, bounds=bounds)


    for _, strip in strips.iterrows():

        process_strip(strip=strip, is2_data=is2_data)
        return

        


    

