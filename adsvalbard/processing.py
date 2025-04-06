from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.spatial
import shapely.geometry
import xarray as xr
import xdem
from tqdm import tqdm
from variete import VRaster

import adsvalbard.arcticdem
import adsvalbard.icesat2
import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS

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
    coords = []
    for geom in dem_data.geometry.geoms:
        for i in range(100):
            point = geom.exterior.interpolate(i / 100, normalized=True)
            coords.append([point.x, point.y])

    coords = np.array(coords)
    coords -= np.mean(coords, axis=0)[None, :]

    # mean_angle = np.deg2rad(np.atan(dem_data["pgc:avg_convergence_angle"]))

    is2 = adsvalbard.icesat2.filter_is2_data(
        bounds=dem.bounds,
        dem_data=dem_data,
        is2_data=is2_data,
        cache_label=dem_data["title"],
    ).rename(columns={"h_te_best_fit": "z", "easting": "E", "northing": "N"})

    coreg = xdem.coreg.GradientDescending()

    dem_in_memory = xdem.DEM.from_array(
        dem.read(1), transform=dem.transform, crs=dem.crs, nodata=-9999.0
    )

    is2 = is2[dem_in_memory.value_at_coords(is2["E"], is2["N"]) != -9999.0]

    assert is2[["E", "N"]].notnull().all().all()

    # plt.imshow(dem_in_memory.data, extent=[dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top])
    # plt.scatter(is2["E"], is2["N"])
    # plt.show()

    coreg.fit_pts(is2, dem_in_memory, mask_high_curv=False)

    dem_coreg = coreg.apply(dem_in_memory)

    is2["dem_elev"] = dem_coreg.value_at_coords(is2["E"], is2["N"])

    is2["dh"] = is2["dem_elev"] - is2["z"]

    is2 = is2[is2["dh"] > -8000]

    print(np.median(np.abs(is2["dh"])))

    plt.scatter(is2["E"], is2["N"], c=is2["dh"], vmin=-20, vmax=20, cmap="RdBu")
    plt.show()

    dependent = "z"

    for i, data in is2.groupby(
        (
            20
            * (is2[dependent] - is2[dependent].min())
            / (is2[dependent].max() - is2[dependent].min())
        ).astype(int)
    ):
        plt.boxplot(data["dh"], positions=[i], widths=[1])

    # plt.scatter(is2["N"], is2["dh"])
    # plt.colorbar()
    plt.show()


def process_strip(
    strip: pd.Series, is2_data: xr.Dataset, progress_bar: tqdm | None = None
):
    if progress_bar is not None:
        progress_bar.set_description("Downloading DEM")
    dem_path, mask_path = adsvalbard.arcticdem.download_arcticdem(strip)

    dem = adsvalbard.arcticdem.get_warped_masked_vrt(
        dem_path=dem_path, mask_path=mask_path, res=CONSTANTS.coreg_res
    )

    if progress_bar is not None:
        progress_bar.set_description("Co-registering to IS2")

    coregister_is2(dem, dem_data=strip, is2_data=is2_data)


def process(region: str = "nordenskiold", progress_bar: bool = True):
    bounds = adsvalbard.utilities.get_bounds(region=region)
    strips = adsvalbard.arcticdem.get_strips(region_label=region)

    poi = shapely.geometry.box(*list(bounds))
    strips.sort_values("start_datetime", ascending=False, inplace=True)

    failures = get_previous_failures()

    strips = strips[~strips["title"].isin(failures["title"])]

    is2_data = adsvalbard.icesat2.get_is2_data(region_label=region, bounds=bounds)

    for _, strip in strips.iterrows():
        process_strip(strip=strip, is2_data=is2_data)
        return
