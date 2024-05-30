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
import dask.array as da
import dask
import tqdm.dask
import tqdm.contrib.concurrent
import contextlib
import threading
import functools

import shapely
import sympy
import sympy.abc

import adsvalbard.rasters
import adsvalbard.utilities

import main


def make_dem_stack_mem(
    arcticdem_paths: list[Path],
    chunk_bounds: rio.coords.BoundingBox,
    res: list[float],
    locks: dict[Path, threading.Lock],
):
    shape = (
        int((chunk_bounds.top - chunk_bounds.bottom) / res[0]),
        int((chunk_bounds.right - chunk_bounds.left) / res[1]),
    )

    xr_coords = [
        ("y", np.linspace(chunk_bounds.bottom + res[1] / 2, chunk_bounds.top - res[1] / 2, shape[0])[::-1]),
        ("x", np.linspace(chunk_bounds.left + res[0] / 2, chunk_bounds.right - res[0] / 2, shape[1])),
    ]

    # data = xr.Dataset(coords=xr_coords)
    # with rio.open("temp.heerland/npi_mosaic_clip.tif") as raster:
    #     window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)

    #     data = xr.DataArray(
    #         data=raster.read(1, window=window, boundless=True, masked=True).filled(0.0), coords=xr_coords
    #     ).to_dataset(name="npi_dem")

    # with rio.open("temp.heerland/npi_mosaic_clip_years.tif") as raster:
    #     window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)
    #     data["npi_dem_year"] = ("y", "x"), raster.read(1, window=window, boundless=True, masked=True).filled(2010)

    dems = {}
    nmads = {}
    # for filepath in Path("temp.heerland/arcticdem_coreg/").glob("*.tif"):
    for filepath in arcticdem_paths:
        date = datetime.datetime.strptime(filepath.name.split("_")[3], "%Y%m%d")
        with rio.open(filepath) as raster:
            if not adsvalbard.utilities.bounds_intersect(chunk_bounds, raster.bounds):
                continue

            window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)

            with locks[filepath]:
                dem = raster.read(1, window=window, boundless=True, masked=True).filled(np.nan)

            if np.all(~np.isfinite(dem)):
                continue

            for _ in range(60):
                if date in dems:
                    date = date + datetime.timedelta(seconds=1)
                else:
                    break

            dems[date] = dem

    data = (
        xr.DataArray(list(dems.values()), coords=[("time", list(dems.keys()))] + xr_coords, name="arcticdem")
        .sortby("time")
        .to_dataset()
    )

    return data


def make_dem_stack(
    stack_filepath: Path,
    arcticdem_paths: list[Path],
    chunk_bounds: rio.coords.BoundingBox,
    res: list[float],
    locks: dict[Path, threading.Lock],
) -> Path:
    if stack_filepath.is_file():
        return stack_filepath

    data = make_dem_stack_mem(arcticdem_paths=arcticdem_paths, chunk_bounds=chunk_bounds, res=res, locks=locks)

    data.to_netcdf(stack_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in data.data_vars})

    return stack_filepath


# Very slow but very good interpolation approach
def interp(data: xr.Dataset, degree: int, random_state: int = 0, slope_corr: str = "per-timestamp"):
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


    # data["diff_res"] = data["diff"] - data["diff"].median("time")

    # res = data["diff_res"].to_dataframe().reset_index()

    # slope_corr = "No"
    for col in ["y_slope", "x_slope"]:

        data[col] = "time", np.zeros(data["time"].shape[0], dtype=float)

    if slope_corr == "per-timestamp":

        for time in data["time"].values:
            subset = data.sel(time=time).to_dataframe().reset_index()

            model = sklearn.linear_model.HuberRegressor(fit_intercept=False)
            

            model.fit(subset[["x", "y"]], subset["diff"])


            data["x_slope"].loc[{"time": time}] = model.coef_[0]
            data["y_slope"].loc[{"time": time}] = model.coef_[1]


        #     # model.fit(
        # trend_y = data["diff"].polyfit("y", deg=1)["polyfit_coefficients"].sel(degree=1)
        # print(data)
        # print(trend_y)
        # data["diff"] -= trend_y * data["y"]
        # data["diff"] -= data["diff"].polyfit("x", deg=1)["polyfit_coefficients"].sel(degree=1) * data["x"]
    elif slope_corr == "inter-timestamp":
        median_diff = data["diff"].median().item()
        model = sklearn.linear_model.LinearRegression().fit(res[["y", "x"]].values, res["diff_res"].values)
        yy, xx = xr.broadcast(data["y"], data["x"])
        # print(data["y"].broadcast_like(data["npi_dem"], ))
        data["corr"] = ("y", "x"), model.predict(np.transpose([yy.values.ravel(), xx.values.ravel()])).reshape((data["y"].shape[0], data["x"].shape[0]))
        data["diff"] += data["corr"]

        data["diff"] -= data["diff"].median().item() - median_diff


    data["diff"] = data["diff"] - (data["x_slope"] * data["x"]) - (data["y_slope"] * data["y"])

    data = data.stack(xyt=["time", "y", "x"], create_index=False).swap_dims(xyt="time")
    data = data.dropna("time")

    huber = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)

        model =sklearn.linear_model.HuberRegressor(fit_intercept=False) if huber else (
            sklearn.linear_model.RANSACRegressor(
                estimator=sklearn.linear_model.LinearRegression(fit_intercept=False), random_state=random_state
            )
        )
        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(degree=degree,include_bias=False),
            model,
            # sklearn.linear_model.HuberRegressor(fit_intercept=False)
        )

        model.fit(data["time"].values[:, None], data["diff"].values)


    if huber:
        inlier_mask = np.isfinite(data["diff"].values)
    else:
        inlier_mask = model.steps[-1][-1].inlier_mask_
    score = model.score(data["time"].values[inlier_mask, None], data["diff"].values[inlier_mask])

    out = data[[]]

    # Use the sympy Taylor shift to set x_0 to year zero.
    if huber:
        coefs = model.steps[-1][-1].coef_[::-1]
    else:
        coefs = model.steps[-1][-1].estimator_.coef_[::-1]
    poly = sympy.Poly(coefs, sympy.abc.x)
    poly = poly.shift(-data["npi_dem_year"].item())
    coefs = poly.coeffs()[::-1]

    coefs[0] += data["npi_dem"].item()

    out["y"] = mid_y
    out["x"] = mid_x
    # out = out.expand_dims({"degree": len(coefs)}).assign_coords({"degree": np.arange(len(coefs))})
    out.coords["degree"] = "degree", np.arange(len(coefs))

    out["coefs"] = "degree", coefs
    out["n_inliers"] = np.count_nonzero(inlier_mask)
    out["score"] = score

    out["coefs"] = out["coefs"].astype(float)

    res = model.predict(data["time"].values[:, None]) - data["diff"].values
    out["nmad"] = 1.4826 * np.median(np.abs(res - np.median(res)))

    # plt.plot(data["time"], coefs[0] + coefs[1] * (data["time"].values + data["npi_dem_year"].item()) + coefs[2] * (data["time"].values + data["npi_dem_year"].item()) ** 2)
    # plt.plot(data["time"], np.poly1d(coefs[::-1])(data["time"].values + data["npi_dem_year"].item()))
    if True:
        plt.title(f"{slope_corr.capitalize()} slope correction (R² = {score:.2f}, NMAD: {out['nmad'].item():.2f} m)")
        plt.scatter(data["time"] + data["npi_dem_year"].item(), data["diff"])
        xs = np.linspace(0, data["time"].max().item(), num=100)
        plt.scatter(data["npi_dem_year"].item(), 0, marker="x")
        plt.plot(xs + data["npi_dem_year"].item(), model.predict(xs[:, None]), color="black")
        plt.ylim(-7, 1)
        plt.xlabel("Years")
        plt.ylabel("dH since start (m)")
        plt.show()
        # raise NotImplementedError()
    return out

def determine_polynomials(out_filepath: Path, in_filepath: Path, degree: int = 2, slope_corr: str = "per-timestamp"):

    if out_filepath.is_file():
        return out_filepath
    with xr.open_dataset(in_filepath) as data:
        # data = data.isel(x=slice(200, 300), y=slice(200, 300))

        data = data.sel(x=slice(551000, 551600), y=slice(8645377, 8644875))

        # data["npi_dem"].plot()
        # plt.show()
        # return

        # point = data.sel(x=poi.x, y=poi.y, method="nearest")

        # plt.scatter(point["time"], point["arcticdem"])
        # plt.show()

        coarsened = (
            data.coarsen(x=20, y=20)
            .construct(x=["x_coarse", "x"], y=["y_coarse", "y"])
            .stack(coarse=["y_coarse", "x_coarse"])
            .groupby("coarse")
            .map(interp, degree=degree, slope_corr=slope_corr)
            .unstack()
        )
        coarsened["y"] = coarsened["y"].isel(x_coarse=0)
        coarsened["x"] = coarsened["x"].isel(y_coarse=0)
        coarsened = coarsened.swap_dims(y_coarse="y", x_coarse="x").drop(["y_coarse", "x_coarse"])

        year = 2015
        years = np.linspace(year, year, num=1)
        years = xr.DataArray(years, coords=[("year", years)])
        dhdt = xr.polyval(years, derivative(coarsened["coefs"]))

        dhdt.isel(year=0).plot()
        plt.show()


        print(coarsened)

        return
        temp_path = out_filepath.with_suffix(out_filepath.suffix + ".tmp")
        coarsened.to_netcdf(temp_path)

        shutil.move(temp_path,out_filepath)

    return out_filepath


def derivative(coefficients: xr.DataArray, coord_name: str = "degree") -> xr.DataArray:
    out = coefficients.copy()

    degrees = sorted(out[coord_name].values)
    for i, degree in enumerate(degrees):
        if (len(degrees) - 1) == i:
            out.loc[{coord_name: degree}] = 0.0
        else:
            out.loc[{coord_name: degree}] = out.sel(**{"degree": degree + 1}) * (degree + 1)

    return out


def test_derivative():
    vals = xr.DataArray([[5, 4, 3, 2]], coords=[("x", [0.0]), ("degree", [3, 2, 1, 0])])

    deriv = derivative(vals)

    assert np.array_equal(deriv.values.ravel(), np.array([0, 15, 8, 3]))


def determine_trends(
    out_filepath: Path, in_filepath: Path | xr.Dataset, min_count_per_year: int = 3, chunk_name: str = "", write_lock = None,
):
    if out_filepath.is_file():
        return out_filepath

    with contextlib.ExitStack() as stack:
        if isinstance(in_filepath, xr.Dataset):
            data = in_filepath
        else:
            data = stack.enter_context(xr.open_dataset(in_filepath, chunks="auto"))

        coarse = data.coarsen(x=10, y=10)
        data = coarse.median()

        grouped = data["arcticdem"].groupby("time.year")
        data["yearly_med"] = grouped.median()
        data["yearly_std"] = grouped.std()
        data["yearly_count"] = grouped.count()

        data = data.reindex(indexers={"year": np.arange(2010, 2023)})

        def get_minmax_year_range(values, years: np.ndarray):
            finites = np.isfinite(values)

            if da.count_nonzero(values) < 2:
                return [[[0, 0]]]
            min_idx = np.argmax(finites, axis=0)

            max_idx = values.shape[0] - np.argmax(finites[::-1], axis=0) - 1

            return [years[min_idx], years[max_idx]]

        linear = data["yearly_med"].polyfit("year", deg=1)
        med = data["yearly_med"].median("year")
        res = xr.polyval(data["year"], linear["polyfit_coefficients"]) - med
        data["nmad"] = 1.4826 * (np.abs(res.median("year") - res)).median("year")
        # med = data["yearly_med"].median("year")
        # res = data["yearly_med"] - med

        # data["nmad"] = 1.4826 * (np.abs(res.median("year") - res)).median("year")

        # Use these data for the polynomial fitting:
        # - After 2013 (2012 is generally quite bad)
        # - If the count per year exceeds this value
        # - If the count per year is above 4 (if not, check that it's not too far from 3x the NMAD) 
        data["use_for_poly"] = (
            (data["year"].broadcast_like(data["yearly_med"]) >= 2013)
            (data["yearly_count"] >= min_count_per_year)
            & ~((np.abs(res) > (3 * data["nmad"]).clip(min=5)) & (data["yearly_count"] < 4))
        )

        minmax = xr.apply_ufunc(
            get_minmax_year_range,
            data["use_for_poly"],
            input_core_dims=[["year", "y", "x"]],
            output_core_dims=[["minmax", "y", "x"]],
            dask="allowed",
            kwargs={"years": data["year"].values},
            dask_gufunc_kwargs={"output_sizes": {"minmax": 2}},
            output_dtypes=[data["year"].values.dtype],
        )
        data["min_poly_year"] = minmax.isel(minmax=0).clip(min=2013)
        data["max_poly_year"] = minmax.isel(minmax=1)

        data["chunk_name"] = (
            ("y", "x"),
            np.repeat([chunk_name], (data["y"].shape[0] * data["x"].shape[0])).reshape(
                data["y"].shape[0], data["x"].shape[0]
            ),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.RankWarning)
            res = data["yearly_med"].where(data["use_for_poly"]).polyfit("year", deg=3, full=True)

        data["polyfit_coefficients"] = res["polyfit_coefficients"]

        temp_filepath = out_filepath.with_name(out_filepath.name + ".tmp")
        vars = ["min_poly_year", "max_poly_year", "yearly_med", "yearly_std", "yearly_count", "polyfit_coefficients", "nmad", "use_for_poly"]

        if write_lock is not None:
            stack.enter_context(write_lock)
        data[vars].to_netcdf(temp_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in vars})
        shutil.move(temp_filepath, out_filepath)
        return out_filepath


def process_chunk(
    parameters: dict[str, object],
    locks: dict[Path, threading.Lock],
    arcticdem_paths: list[Path],
    res: tuple[float, float],
    write_lock = None,
):
    data = make_dem_stack_mem(
        arcticdem_paths=arcticdem_paths, chunk_bounds=parameters["chunk_bounds"], res=res, locks=locks
    )
    determine_trends(parameters["coef_filepath"], data, chunk_name=parameters["chunk_id"], write_lock=write_lock)

    return parameters["coef_filepath"]


def stack_dems(stack_res: float = 50.0, chunk_size: int = 1000):
    bounds = main.get_bounds(region="heerland")
    res = main.get_res()

    print("Starting")
    out_bounds = adsvalbard.utilities.align_bounds(bounds, [stack_res] * 2)
    chunks = adsvalbard.rasters.generate_raster_chunks(out_bounds, res=res[0], chunksize=1000)

    arcticdem_paths = list(Path("temp.heerland/arcticdem_coreg").glob("*.tif"))
    locks = {path: threading.Lock() for path in arcticdem_paths}

    filepaths = []
    to_process = []
    for i, chunk_bounds in enumerate(chunks):

        chunk_id = str(i).zfill(3)
        coef_filepath = Path(f"temp/coeffs_chunk_{chunk_id}.nc")

        chunk_bnd_box = shapely.geometry.box(*chunk_bounds)
        poi = shapely.geometry.Point(550030, 8644360)
        if not chunk_bnd_box.contains(poi):
            continue
        stack_filepath = Path(f"temp/dem_stack_chunk_{chunk_id}.nc")
        make_dem_stack(stack_filepath, arcticdem_paths=arcticdem_paths, chunk_bounds=chunk_bounds, locks=locks, res=res)

        poly_filepath = Path(f"temp/robust_coeffs_chunk_{chunk_id}.nc")

        determine_polynomials(poly_filepath, stack_filepath)

        return


        if not coef_filepath.is_file():
            to_process.append({"chunk_bounds": chunk_bounds, "chunk_id": chunk_id, "coef_filepath": coef_filepath})
        filepaths.append(coef_filepath)

    if len(to_process) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)
            # Netcdf cannot do concurrent writes, so this write lock is needed...
            write_lock =threading.Lock()
            tqdm.contrib.concurrent.thread_map(
                functools.partial(process_chunk, locks=locks, res=res, arcticdem_paths=arcticdem_paths, write_lock=write_lock),
                to_process,
                desc="Processing chunks",
                max_workers=3,
            )

    # for i, chunk_bounds in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
    #     chunk_bnd_box = shapely.geometry.box(*chunk_bounds)

    #     poi = shapely.geometry.Point(550030, 8644360)

    #     # if not chunk_bnd_box.contains(poi):
    #     #     continue
    #     chunk_id = str(i).zfill(3)

    #     filepath = Path(f"temp/dem_stack_chunk_{chunk_id}.nc")
    #     coef_filepath = filepath.with_stem(filepath.stem.replace("dem_stack", "coeffs"))

    #     # print("Making DEM stack")
    #     # make_dem_stack(filepath, chunk_bounds=chunk_bounds, res=res)

    #     # determine_trends(filepath)
    #     if not coef_filepath.is_file():
    #         data = make_dem_stack_mem(arcticdem_paths=arcticdem_paths, chunk_bounds=chunk_bounds, res=res, locks=locks)
    #         determine_trends(coef_filepath, data, chunk_name=chunk_id)

    #     filepaths.append(coef_filepath)

    with xr.open_mfdataset(filepaths, chunks="auto") as data:
        # data["polyfit_coefficients"] = data["polyfit_coefficients"].isel(year=0)

        # def get_minmax_year_range(values, years: np.ndarray):
        #     finites = np.isfinite(values)

        #     if da.count_nonzero(values) < 2:
        #         return [[[0, 0]]]
        #     min_idx = np.argmax(finites, axis=0)

        #     max_idx = values.shape[0] - np.argmax(finites[::-1], axis=0) - 1

        #     out = da.zeros(shape=(2,) + min_idx.shape, chunks=((2,),) + min_idx.chunks)
        #     out[0, :, :] = years[min_idx]
        #     out[1, :, :] = years[max_idx]

        #     return out
        #     # return [years[min_idx], years[max_idx]]

        # minmax = xr.apply_ufunc(get_minmax_year_range, data["yearly_med"], input_core_dims=[["year", "y", "x"]], output_core_dims=[["minmax", "y", "x"]], dask="allowed", kwargs={"years": data["year"].values}, dask_gufunc_kwargs={"output_sizes": {"minmax": 2}}, output_dtypes=[data["year"].values.dtype])

        # point = data.sel(x=[564212], y=[8623608], method="nearest").compute()
        # data["med"] = data["yearly_med"].median("year")
        # data["med_res"] = data["
        # med = data["yearly_med"].median("year")
        # res = data["yearly_med"] - med

        # data["polyfit_coefficients"] = data["yearly_med"].where((res < (3 * data["nmad"])) & (data["yearly_count"] >= 3) & (data["year"] >= 3)).polyfit("year", deg=3)["polyfit_coefficients"]
        # point = data.sel(x=550473, y=8641865, method="nearest")

        d0 = data.sel(year=slice(2012, 2017))
        d1 = data.sel(year=slice(2017, 2023))

        diff0 = d0["yearly_med"].diff("year").median("year")
        diff1 = d1["yearly_med"].diff("year").median("year")

        diff0.plot(vmin=-10, vmax=10, cmap="RdBu")
        plt.show()
        diff1.plot(vmin=-10, vmax=10, cmap="RdBu")
        plt.show()

        return
        (diff1 - diff0).plot(vmin=-10, vmax=10, cmap="RdBu")

        # data["yearly_med"].diff("year").median("year").plot(vmin=-10, vmax=10, cmap="RdBu")
        plt.show()

        return

        if False:
            linear = data["yearly_med"].polyfit("year", deg=1)
            med = data["yearly_med"].median("year")
            res = xr.polyval(data["year"], linear["polyfit_coefficients"]) - med
            # # data["nmad"] = 1.4826 * (np.abs(res.median("year") - res)).median("year")
            data["use_for_poly"] = (
                # (data["year"].broadcast_like(data["yearly_med"]) >= 2012)
                 (data["yearly_count"] > 3)
                & ~((np.abs(res) > (3 * data["nmad"]).clip(min=5)) & (data["yearly_count"] < 4))
            )

        # deg = 2
        # data["polyfit_coefficients"] = data["yearly_med"].where(data["use_for_poly"]).polyfit("year", deg=deg)["polyfit_coefficients"]
        # if deg == 2:
        #     data["polyfit_coefficients"].loc[{"degree": 3}] = 0

        points = {
            "morsnev_front": {"x": 563416, "y": 8612261},
            "scheele_front": {"x": 549132, "y": 8633176},
            "morsnev_upper": {"x": 564529, "y": 8625499},
        }

        for i, key in enumerate(points):
            plt.subplot(1, len(points), i + 1)
            point = data.sel(**points[key], method="nearest").compute()
            point["yearly_med"].plot()
            plt.errorbar(point["year"], point["yearly_med"], point["yearly_std"])
            print(point["polyfit_coefficients"])
            poly = xr.polyval(point["year"], point["polyfit_coefficients"])
            print(poly)
            poly.plot()
            plt.title(key)
            for year in point["year"].values:
                yearly = point.sel(year=year)
                plt.annotate(f'{yearly["yearly_count"].item()}', (year, yearly["yearly_med"].item()))
        plt.show()

        # return
        point = data.sel(x=561886, y=8650826, method="nearest")
        # Morsnevbreen
        # point = data.sel(x=567974, y=8643688, method="nearest")


        # print(point)
        # return

        # point["min_year"] = minmax.isel(minmax=0)
        # point["max_year"] = minmax.isel(minmax=1)

        # point = point.isel(x=0, y=0).compute()

        # point["yearly_med"].plot()
        # xr.polyval(years, point["polyfit_coefficients"]).plot()
        # ylim = plt.gca().get_ylim()
        # plt.vlines([point["min_poly_year"], point["max_poly_year"]], *ylim)
        # plt.ylim(ylim)
        # plt.show()

        # print(point)
        # return
        # data["yearly_med"].sel(year=2021).plot()
        # plt.show()
        # print(data)
        # return

        year = 2015
        years = np.linspace(year, year, num=1)
        years = xr.DataArray(years, coords=[("year", years)])
        dhdt = xr.polyval(years, derivative(data["polyfit_coefficients"]))
        print(dhdt)

        dhdt.isel(year=-1).plot(vmin=-10, vmax=10, cmap="RdBu")
        plt.show()
        return

        for year in years["year"].values:
            plt.figure(figsize=(7, 5))
            # mask = ((data["min_poly_year"].astype(float) >= year) & (data["max_poly_year"].astype(float) <= year)).astype(float)

            # mask=(data["min_poly_year"].astype(float) >= year)# & (data["max_poly_year"].astype(float) >= year)
            # mask = mask.where(mask == 1)
            dhdt.sel(year=year).plot(vmin=-10, vmax=10, cmap="RdBu")
            # mask.plot(vmin=0, vmax=2, cmap="Greys")
            year_str = str(round(year * 100) / 100)[::-1].zfill(7)[::-1]
            plt.savefig(f"figures/dhdt/dhdt_{year_str}.jpg", dpi=300)
        print(dhdt)
        return

        plt.show()


if __name__ == "__main__":
    stack_dems()
