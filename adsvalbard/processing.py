import adsvalbard.utilities
import adsvalbard.arcticdem
import adsvalbard.icesat2
import shapely.geometry
from adsvalbard.constants import CONSTANTS
import pandas as pd
from tqdm import tqdm

FAILURE_FILE = CONSTANTS.temp_dir.joinpath("failures.csv")

def register_failure(dem_id: str, exception: str):
    FAILURE_FILE.parent.mkdir(exist_ok=True)
    with open(FAILURE_FILE, "a+") as outfile:
        outfile.write(f'"{dem_id}","{exception}"\n')

def get_previous_failures() -> pd.DataFrame:
    if FAILURE_FILE.is_file():
        return pd.read_csv(FAILURE_FILE, names=["title", "exception"])
    return pd.DataFrame(columns=["title", "exception"])

  
def process(region: str = "nordenskiold", progress_bar: bool = True):

    bounds = adsvalbard.utilities.get_bounds(region=region)
    strips = adsvalbard.arcticdem.get_strips(cache_label=region)

    poi = shapely.geometry.box(*list(bounds))
    strips.sort_values("start_datetime", ascending=False, inplace=True)

    failures = get_previous_failures()

    strips = strips[~strips["title"].isin(failures["title"])]

    ic2 = adsvalbard.icesat2.get_ic2_data(cache_label=region, bounds=bounds)

    print(ic2)

    

