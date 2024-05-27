import rasterio as rio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import xarray as xr
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.exceptions
import warnings
import shutil

import shapely
import sympy
import sympy.abc

import adsvalbard.rasters
import adsvalbard.utilities

import main


def make_dem_stack(stack_filepath: Path, chunk_bounds: rio.coords.BoundingBox, res: list[float]):
    if stack_filepath.is_file():
        return stack_filepath

    shape = (
        int((chunk_bounds.top - chunk_bounds.bottom) / res[0]),
        int((chunk_bounds.right - chunk_bounds.left) / res[1]),
    )

    xr_coords = [
        ("y", np.linspace(chunk_bounds.bottom + res[1] / 2, chunk_bounds.top - res[1] / 2, shape[0])[::-1]),
        ("x", np.linspace(chunk_bounds.left + res[0] / 2, chunk_bounds.right - res[0] / 2, shape[1])),
    ]

    with rio.open("temp.heerland/npi_mosaic_clip.tif") as raster:
        window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)

        data = xr.DataArray(
            data=raster.read(1, window=window, boundless=True, masked=True).filled(0.0), coords=xr_coords
        ).to_dataset(name="npi_dem")

    with rio.open("temp.heerland/npi_mosaic_clip_years.tif") as raster:
        window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)
        data["npi_dem_year"] = ("y", "x"), raster.read(1, window=window, boundless=True, masked=True).filled(2010)

    dems = {}
    for filepath in Path("temp.heerland/arcticdem_coreg/").glob("*.tif"):
        date = datetime.datetime.strptime(filepath.name.split("_")[3], "%Y%m%d")
        with rio.open(filepath) as raster:
            if not adsvalbard.utilities.bounds_intersect(chunk_bounds, raster.bounds):
                continue

            window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)

            dem = raster.read(1, window=window, boundless=True, masked=True).filled(np.nan)

            if np.all(~np.isfinite(dem)):
                continue

            for _ in range(60):
                if date in dems:
                    date = date + datetime.timedelta(seconds=1)
                else:
                    break

            dems[date] = dem

    data["arcticdem"] = xr.DataArray(
        list(dems.values()), coords=[("time", list(dems.keys()))] + xr_coords, name="arcticdem"
    ).sortby("time")

    data.to_netcdf(stack_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in data.data_vars})

    return stack_filepath


def interp(data: xr.Dataset, random_state: int = 0, degree=2):
    mid_y = data["y"].mean("y").item()
    mid_x = data["x"].mean("x").item()

    data["y"] -= mid_y
    data["x"] -= mid_x

    data["diff"] = data["arcticdem"] - data["npi_dem"]

    # data = data.median(["y", "x"])
    data[["npi_dem", "npi_dem_year"]] = data[["npi_dem", "npi_dem_year"]].median(["y", "x"])

    data.coords["time"] = (
        data["time"].dt.year
        + data["time"].dt.month / 12
        + data["time"].dt.day / (30 * 12)
        - data["npi_dem_year"]
        + np.linspace(0, 1e-9, data["time"].shape[0])
    )

    data = data.dropna("time")

    trend_y = data["diff"].polyfit("y", deg=1)["polyfit_coefficients"].isel(degree=0) * data["y"]
    data["diff"] -= trend_y
    data["diff"] -= data["diff"].polyfit("x", deg=1)["polyfit_coefficients"].isel(degree=0) * data["x"]

    data = data.stack(xyt=["time", "y", "x"], create_index=False).swap_dims(xyt="time")
    data = data.dropna("time")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(degree=degree),
            sklearn.linear_model.RANSACRegressor(
                estimator=sklearn.linear_model.LinearRegression(fit_intercept=False), random_state=random_state
            ),
        )

        model.fit(data["time"].values[:, None], data["diff"].values)


    inlier_mask = model.steps[-1][-1].inlier_mask_
    score = model.score(data["time"].values[inlier_mask, None], data["diff"].values[inlier_mask])

    out = data[[]].drop(["degree"])

    # Use the sympy Taylor shift to set x_0 to year zero.
    coefs = model.steps[-1][-1].estimator_.coef_[::-1]
    poly = sympy.Poly(coefs, sympy.abc.x)
    poly = poly.shift(-data["npi_dem_year"].item())
    coefs = poly.coeffs()[::-1]

    coefs[0] += data["npi_dem"].item()

    out["y"] = mid_y
    out["x"] = mid_x
    out.coords["degree"] = "degree", np.arange(len(coefs))

    out["coefs"] = "degree", coefs
    out["n_inliers"] = np.count_nonzero(inlier_mask)
    out["score"] = score

    out["coefs"] = out["coefs"].astype(float)

    res = model.predict(data["time"].values[:, None]) - data["diff"].values
    out["nmad"] = 1.4826 * np.median(np.abs(res - np.median(res)))

    # plt.plot(data["time"], coefs[0] + coefs[1] * (data["time"].values + data["npi_dem_year"].item()) + coefs[2] * (data["time"].values + data["npi_dem_year"].item()) ** 2)
    # plt.plot(data["time"], np.poly1d(coefs[::-1])(data["time"].values + data["npi_dem_year"].item()))
    # plt.scatter(data["time"], data["arcticdem"])
    # # plt.plot(data["time"], model.predict(data["time"].values[:, None]))
    # # plt.ylim(-8, 0)
    # plt.show()
    return out


def determine_polynomials(out_filepath: Path, in_filepath: Path, degree: int = 2):

    if out_filepath.is_file():
        return out_filepath
    with xr.open_dataset(in_filepath) as data:
        # data = data.isel(x=slice(200, 300), y=slice(200, 300))

        # point = data.sel(x=poi.x, y=poi.y, method="nearest")

        # plt.scatter(point["time"], point["arcticdem"])
        # plt.show()

        coarsened = (
            data.coarsen(x=10, y=10)
            .construct(x=["x_coarse", "x"], y=["y_coarse", "y"])
            .stack(coarse=["y_coarse", "x_coarse"])
            .groupby("coarse")
            .map(interp, degree=degree)
            .unstack()
        )
        coarsened["y"] = coarsened["y"].isel(x_coarse=0)
        coarsened["x"] = coarsened["x"].isel(y_coarse=0)
        coarsened = coarsened.swap_dims(y_coarse="y", x_coarse="x").drop(["y_coarse", "x_coarse"])

        temp_path = out_filepath.with_suffix(out_filepath.suffix + ".tmp")
        coarsened.to_netcdf(temp_path)

        shutil.move(temp_path,out_filepath)

    return out_filepath

    

def stack_dems(stack_res: float = 50.0, chunk_size: int = 1000):
    bounds = main.get_bounds(region="heerland")
    res = main.get_res()

    out_bounds = adsvalbard.utilities.align_bounds(bounds, [stack_res] * 2)
    chunks = adsvalbard.rasters.generate_raster_chunks(out_bounds, res=res[0], chunksize=1000)

    for i, chunk_bounds in enumerate(chunks):
        chunk_bnd_box = shapely.geometry.box(*chunk_bounds)

        poi = shapely.geometry.Point(550030, 8644360)

        if not chunk_bnd_box.contains(poi):
            continue

        filepath = Path(f"temp/dem_stack_chunk_{str(i).zfill(3)}.nc")
        coef_filepath = filepath.with_stem(filepath.stem.replace("dem_stack", "coeffs"))

        make_dem_stack(filepath, chunk_bounds=chunk_bounds, res=res)

        determine_polynomials(coef_filepath, filepath)

        with xr.open_dataset(coef_filepath) as data:

            yr = 2022
            data["elev1"] = data["coefs"].sel(degree=0) + data["coefs"].sel(degree=1) * yr + data["coefs"].sel(degree=2) * yr ** 2 
            yr = 2010
            data["elev0"] = data["coefs"].sel(degree=0) + data["coefs"].sel(degree=1) * yr + data["coefs"].sel(degree=2) * yr ** 2
            # data["coefs"].sel(degree=1).plot()
            # data["diff"] = data["elev1"] - data["elev0"]
            data["coefs"].sel(degree=2).plot()
            plt.show()


if __name__ == "__main__":
    stack_dems()
