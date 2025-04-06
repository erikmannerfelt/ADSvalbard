import contextlib
import datetime
import functools
import shutil
import threading
import time
import warnings
from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import sklearn.exceptions
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sympy
import sympy.abc
import tqdm.contrib.concurrent
import tqdm.dask
import xarray as xr

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

    dem_path = next(filter(lambda fp: "npi_mosaic_clip.tif" in fp.name, locks.keys()))
    dem_years_path = next(filter(lambda fp: "clip_years.tif" in fp.name, locks.keys()))
    # dem_path = ["npi_mosaic_clip.tif" in fp.name for fp in locks.keys()][0]
    # dem_years_path = ["clip_years.tif" in fp.name for fp in locks.keys()][0]
    # data = xr.DataArray(coords=xr_coords)
    with rio.open(dem_path) as raster:
        window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)

        with locks[dem_path]:
            data = xr.DataArray(
                data=raster.read(1, window=window, boundless=True, masked=True).filled(0.0), coords=xr_coords
            ).to_dataset(name="npi_dem")

    with rio.open(dem_years_path) as raster:
        window = rio.windows.from_bounds(*chunk_bounds, transform=raster.transform)
        with locks[dem_years_path]:
            data["npi_dem_year"] = ("y", "x"), raster.read(1, window=window, boundless=True, masked=True).filled(2010)

    dems = {}
    nmads = {}
    # for filepath in Path("temp.heerland/arcticdem_coreg/").glob("*.tif"):
    for filepath in arcticdem_paths:
        if "npi_" in filepath.name:
            continue
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

    data["arcticdem"] = (
        xr.DataArray(list(dems.values()), coords=[("time", list(dems.keys()))] + xr_coords, name="arcticdem")
        .sortby("time")
        # .to_dataset()
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
    for col in ["y_slope", "x_slope"]:
        data[col] = "time", np.zeros(data["time"].shape[0], dtype=float)

    for time_coord in data["time"].values:
        subset = data.sel(time=time_coord).to_dataframe().reset_index()

        model = sklearn.linear_model.HuberRegressor(fit_intercept=False)

        model.fit(subset[["x", "y"]], subset["diff"])

        data["x_slope"].loc[{"time": time_coord}] = model.coef_[0]
        data["y_slope"].loc[{"time": time_coord}] = model.coef_[1]

    data["diff"] = data["diff"] - (data["x_slope"] * data["x"]) - (data["y_slope"] * data["y"])

    data = data.stack(xyt=["time", "y", "x"], create_index=False).swap_dims(xyt="time")
    data = data.dropna("time")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)

        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(degree=degree, include_bias=False),
            sklearn.linear_model.RANSACRegressor(
                estimator=sklearn.linear_model.LinearRegression(fit_intercept=False), random_state=random_state
            ),
        )

        model.fit(data["time"].values[:, None], data["diff"].values)

    inlier_mask = model.steps[-1][-1].inlier_mask_
    score = model.score(data["time"].values[inlier_mask, None], data["diff"].values[inlier_mask])

    out = data[[]]

    # Use the sympy Taylor shift to set x_0 to year zero.
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


def get_poly_inner(arr: da.Array, res: float, orig_shape: list[int], times: np.ndarray, npi_dem_year: float, degree: tuple[int, ...]):

    if arr.shape[0] == 1:
        return np.empty((6 + max(degree) + 1,len(degree)), dtype=np.float64)
    arr = arr.reshape(orig_shape)

    coords = (
        np.repeat(
            np.array(
                np.meshgrid(
                    np.linspace(-(arr.shape[1] / 2) * res, (arr.shape[1] / 2) * res, arr.shape[1]),
                    np.linspace(-(arr.shape[0] / 2) * res, (arr.shape[0] / 2) * res, arr.shape[0])[::-1],
                )
            )[:, :, :, None],
            orig_shape[-1],
            axis=3,
        )
        .reshape((2, -1))
        .T
    )

    arr = arr.reshape((-1, arr.shape[-1]))

    resi = arr - np.nanmedian(arr, axis=0)[None, :]
    resi = resi.ravel()

    finites = np.isfinite(resi)

    model = sklearn.linear_model.HuberRegressor(fit_intercept=False)
    model.fit(coords[finites], resi[finites])

    arr -= model.predict(coords).reshape(arr.shape)

    slope_x, slope_y = model.coef_
    # filled = ~np.all(~np.isfinite(d_h), axis=0)
    # d_t = d_t[:, filled]
    # d_h = d_h[:, filled]

    d_t = np.repeat(times[None, :], arr.shape[0], axis=0).ravel()
    arr = arr.ravel()

    filled = np.isfinite(arr)
    d_t = d_t[filled]
    arr = arr[filled]

    out = []
    for deg in degree:

        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(degree=deg, include_bias=False),
            sklearn.linear_model.RANSACRegressor(
                estimator=sklearn.linear_model.LinearRegression(fit_intercept=False),
                max_trials=100,
                stop_probability=0.7,
                min_samples=4,
                random_state=0,
            ),
        )

        model.fit(d_t[:, None], arr)

        inlier_mask = model.steps[-1][-1].inlier_mask_

        score = model.score(d_t[inlier_mask, None], arr[inlier_mask])
        pred = model.predict(d_t[:, None])

        n_inliers = np.count_nonzero(inlier_mask)
        n_total = inlier_mask.size

        resi = arr - pred
        nmad = 1.4826 * np.median(np.abs(np.median(resi) - resi))

        # # Use the sympy Taylor shift to set x_0 to year zero.
        # The "[::-1]" are because sklearn and sympy polys are ordered differently:
        # - sympy: ... + ax² + bx + c
        # - sklearn: a + bx + cx² + ...
        coefs_init = np.r_[[0.00000001], model.steps[-1][-1].estimator_.coef_]
        poly = sympy.Poly(coefs_init[::-1], sympy.abc.x).shift(-npi_dem_year)
        coefs = np.r_[poly.all_coeffs()[::-1], [0.] * (max(degree) - deg)]

        # print(coefs)

        if False:
            plt.title(f"NMAD: {nmad:.2f} m, score: {score:.2f}")
            plt.scatter(d_t + npi_dem_year, arr, alpha=0.1)

            d_ts = np.linspace(d_t.min(), d_t.max())
            plt.plot(d_ts + npi_dem_year, model.predict(d_ts[:, None]))
            plt.plot(d_ts + npi_dem_year, coefs_init[2] * d_ts**2 + coefs_init[1] * d_ts + coefs_init[0])
            d_ts += npi_dem_year
            plt.plot(d_ts, coefs[2] * d_ts**2 + coefs[1] * d_ts + coefs[0])

            plt.ylim(pred.min() - 0.5, pred.max() + 0.5)
            plt.show()

        # print(score)

        out.append(np.r_[coefs, [nmad, score, n_inliers, n_total, slope_x, slope_y]])

    return np.transpose(out)

    yvals = arr[:, :, 1, :]
    print(arr.shape)
    raise NotImplementedError()
    ...


def get_poly(arr: da.Array, res: float, times: np.ndarray, npi_dem_year: float, degree: tuple[int, ...]):
    red = da.apply_along_axis(
        get_poly_inner,
        axis=1,
        arr=arr.reshape((arr.shape[0], -1)),
        res=res,
        orig_shape=arr.shape[1:],
        times=times,
        npi_dem_year=npi_dem_year,
        degree=degree,
    )
    return red


def determine_polynomials(out_filepath: Path, in_data: Path | xr.Dataset, degree: tuple[int, ...] = (1, 2, 3, 4), slope_corr: str = "per-timestamp", write_lock = None, chunk_num: int | None = None):
    if out_filepath.is_file():
        return out_filepath

    with contextlib.ExitStack() as stack:

        if isinstance(in_data, Path):
            data = stack.enter_context(xr.open_dataset(in_data))
        else:
            data = in_data
        # data = data.isel(x=slice(200, 400), y=slice(200, 300)).load()

        # data = data.sel(x=slice(551000, 551600), y=slice(8645377, 8644875))

        # data["npi_dem"].plot()
        # plt.show()
        # return

        # point = data.sel(x=poi.x, y=poi.y, method="nearest")

        # plt.scatter(point["time"], point["arcticdem"])
        # plt.show()
        data.coords["time"] = (
            data["time"].dt.year
            + data["time"].dt.month / 12
            + data["time"].dt.day / (30 * 12)
            + np.linspace(0, 1e-9, data["time"].shape[0])
        )

        coarsened = (
            data.coarsen(x=20, y=20)
            .construct(x=["x_coarse", "x"], y=["y_coarse", "y"])
            .stack(coarse=["y_coarse", "x_coarse"])
        )
        coarsened["d_h"] = coarsened["arcticdem"] - coarsened["npi_dem"]
        coarsened["npi_dem_year"] = coarsened["npi_dem_year"].median(["y", "x"]).compute()

        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


        res = []
        for year, subset in coarsened.groupby("npi_dem_year"):
            subset_res = xr.apply_ufunc(
                get_poly,
                subset["d_h"],
                input_core_dims=[["coarse", "y", "x", "time"]],
                output_core_dims=[["coarse", "retvals", "order"]],
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={
                    "output_sizes": {"retvals": 7 + max(degree),"order": len(degree)},

                },
                kwargs={
                    "res": 5.0,
                    "times": coarsened["time"].values - year,
                    "npi_dem_year": year,
                    "degree": degree
                },
                dask="parallelized",

            )
            res.append(subset_res)

        res = xr.combine_nested(res, "coarse").unstack()

        now = time.time()

        out = coarsened.unstack()[["npi_dem", "npi_dem_year"]]
        grouped = coarsened.unstack()["arcticdem"].groupby(coarsened["time"].astype(int))
        out["ad_yearly_med"] = grouped.median(["time","y", "x"])
        out["ad_yearly_count"] = grouped.count(["time","y", "x"])
        out["ad_yearly_std"] = grouped.std(["time","y", "x"])

        out["y"] = out["y"].isel(x_coarse=0).mean("y")
        out["x"] = out["x"].isel(y_coarse=0).mean("x")
        out["npi_dem"] = out["npi_dem"].median(["y", "x"])

        if chunk_num is not None:
            out["chunk_num"] = ("y", "x"), np.repeat(chunk_num, out["y"].shape[0] * out["x"].shape[0]).reshape((out["y"].shape[0], out["x"].shape[0]))

        out["coefs"] = res.isel(retvals=slice(0, max(degree) + 1)).swap_dims(retvals="degree").assign_coords(degree=np.arange(max(degree) + 1), order=np.array(degree))

        for i, col in enumerate(["nmad", "score", "n_inliers", "n_total", "x_slope", "y_slope"]):
            col_val = res.isel(retvals=max(degree) + 1 + i)
            if "n_" in col:
                col_val = col_val.astype(int)
            out[f"fit_{col}"] = col_val

        out = out.swap_dims(y_coarse="y", x_coarse="x").drop(["x_coarse", "y_coarse"]).rename(time="year")
        out = out.reindex(indexers={"year": np.arange(2010, 2023)})
        out["coefs"].loc[{"degree": 0}] += out["npi_dem"]

        temp_filepath = out_filepath.with_name(out_filepath.name + ".tmp")
        if write_lock is not None:
            stack.enter_context(write_lock)
        out.to_netcdf(temp_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in out.data_vars})
        shutil.move(temp_filepath, out_filepath)
        return out_filepath

        # duration = time.time() - now
        # duration_per_cell = duration / coarsened["coarse"].shape[0]
        # print(f"Duration: {duration:.3}s ({duration_per_cell:.3f} s per cell)")
        # print(coarsened)
        # coarsened["coef"] = 
        # print(coarsened["coef"].isel(coarse=0).compute())

        # coarsened["res"] = res

        # coarsened = coarsened.unstack()

        # print(coarsened)
        # coarsened["res"pe].isel(retvals=-6).plot()
        # plt.show()

        # print(coarsened)
        return

        raise NotImplementedError()

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

        shutil.move(temp_path, out_filepath)

    return out_filepath


def derivative(coefficients: xr.DataArray, coord_name: str = "degree", level: int = 1) -> xr.DataArray:
    # out = coefficients.copy()
    out = xr.zeros_like(coefficients)

    degrees = sorted(out[coord_name].values)
    for i, degree in enumerate(degrees):
        if (len(degrees) - 1) == i:
            out.loc[{coord_name: degree}] = 0.0
        else:
            out.loc[{coord_name: degree}] = coefficients.sel(**{"degree": degree + 1}) * (degree + 1)

    for _ in range(level - 1):
        out = derivative(out, coord_name=coord_name, level=1)

    return out


def test_derivative():
    vals = xr.DataArray([[5, 4, 3, 2]], coords=[("x", [0.0]), ("degree", [3, 2, 1, 0])])

    deriv = derivative(vals)

    assert np.array_equal(deriv.values.ravel(), np.array([0, 15, 8, 3]))


def determine_trends(
    out_filepath: Path,
    in_filepath: Path | xr.Dataset,
    min_count_per_year: int = 3,
    chunk_name: str = "",
    write_lock=None,
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
        data["use_for_poly"] = (data["year"].broadcast_like(data["yearly_med"]) >= 2013)(
            data["yearly_count"] >= min_count_per_year
        ) & ~((np.abs(res) > (3 * data["nmad"]).clip(min=5)) & (data["yearly_count"] < 4))

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
        vars = [
            "min_poly_year",
            "max_poly_year",
            "yearly_med",
            "yearly_std",
            "yearly_count",
            "polyfit_coefficients",
            "nmad",
            "use_for_poly",
        ]

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
    write_lock=None,
):
    data = make_dem_stack_mem(
        arcticdem_paths=arcticdem_paths, chunk_bounds=parameters["chunk_bounds"], res=res, locks=locks
    )
    # determine_trends(parameters["coef_filepath"], data, chunk_name=parameters["chunk_id"], write_lock=write_lock)
    determine_polynomials(parameters["poly_filepath"], data, write_lock=write_lock, chunk_num=parameters["chunk_num"])

    return parameters["poly_filepath"]


def stack_dems(stack_res: float = 50.0, chunk_size: int = 1000):
    bounds = main.get_bounds(region="heerland")
    res = main.get_res()

    print("Starting")
    out_bounds = adsvalbard.utilities.align_bounds(bounds, [stack_res] * 2)
    chunks = adsvalbard.rasters.generate_raster_chunks(out_bounds, res=res[0], chunksize=1000)

    arcticdem_paths = list(Path("temp.heerland/arcticdem_coreg").glob("*.tif"))
    locks = {path: threading.Lock() for path in arcticdem_paths}
    locks[Path("temp.heerland/npi_mosaic_clip.tif")] = threading.Lock()
    locks[Path("temp.heerland/npi_mosaic_clip_years.tif")] = threading.Lock()

    filepaths = []
    to_process = []
    for i, chunk_bounds in enumerate(chunks):
        chunk_id = str(i).zfill(3)
        # coef_filepath = Path(f"temp/coeffs_chunk_{chunk_id}.nc")

        # chunk_bnd_box = shapely.geometry.box(*chunk_bounds)
        # poi = shapely.geometry.Point(550030, 8644360)
        # if not chunk_bnd_box.contains(poi):
        #     continue
        # # stack_filepath = Path(f"temp/dem_stack_chunk_{chunk_id}.nc")

        # # stack = make_dem_stack_mem(arcticdem_paths=arcticdem_paths, chunk_bounds=chunk_bounds, locks=locks, res=res)
        # # make_dem_stack(stack_filepath, arcticdem_paths=arcticdem_paths, chunk_bounds=chunk_bounds, locks=locks, res=res)

        # chunk_id = "temp"
        poly_filepath = Path(f"temp/robust_coeffs_chunk_{chunk_id}.nc")
        parameters = {"chunk_bounds": chunk_bounds, "chunk_id": chunk_id, "poly_filepath": poly_filepath, "chunk_num": i}
        # process_chunk(parameters, locks=locks, arcticdem_paths=arcticdem_paths, res=res)

        # return
        # return

        if not poly_filepath.is_file():
            to_process.append(parameters)
        filepaths.append(poly_filepath)

    if len(to_process) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)
            # Netcdf cannot do concurrent writes, so this write lock is needed...
            write_lock = threading.Lock()
            tqdm.contrib.concurrent.thread_map(
                functools.partial(
                    process_chunk, locks=locks, res=res, arcticdem_paths=arcticdem_paths, write_lock=write_lock
                ),
                to_process,
                desc="Processing chunks",
                max_workers=1,
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

    # with xr.open_mfdataset(filepaths, chunks="auto") as data:
    with contextlib.ExitStack() as stack:

        data_list = []
        for filepath in filepaths:
            part = stack.enter_context(xr.open_dataset(filepath, chunks="auto")).reindex(year=np.arange(2011, 2023))

            data_list.append(part)

        data  =xr.combine_by_coords(data_list)
        # data["fit_nmad"].plot(vmin=0, vmax=5)
        # plt.show()
        # return

        # return

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

        if True:
            points = {
                "morsnev_front": {"x": 563416, "y": 8612261},
                "scheele_front": {"x": 549132, "y": 8633176},
                "morsnev_upper": {"x": 564529, "y": 8625499},
            }

            for i, key in enumerate(points):
                plt.subplot(1, len(points), i + 1)
                point = data.sel(**points[key], method="nearest").compute()
                point["ad_yearly_med"].plot()
                plt.errorbar(point["year"], point["ad_yearly_med"], point["ad_yearly_std"])

                for order, coefs in point["coefs"].groupby("order"):
                    poly = xr.polyval(point["year"], coefs)
                    plt.plot(poly["year"], poly, label=f"{order} degree")
                plt.title(key)
                for year in point["year"].values:
                    yearly = point.sel(year=year)
                    # plt.annotate(f'{yearly["ad_yearly_count"].item()}', (year, yearly["ad_yearly_med"].item()))
                plt.legend()
            plt.show()
            return

        if False:
            for i, (order, poly) in enumerate(data.groupby("order")):
                plt.subplot(2, 2, i + 1)
                polyv = xr.polyval(data["year"], poly["coefs"])

                res = (data["ad_yearly_med"] - polyv).compute()
                offset = 1.4826 * np.abs(res - res.median()).median()

                print(f"{order}: {offset:.2f}")
                plt.title(order)
                poly["fit_nmad"].plot(vmin=0, vmax=2)
            plt.show()
            return

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

        # year = 2015
        data = data.sel(order=2)
        years = np.arange(2013, 2023)
        years = xr.DataArray(years, coords=[("year", years)])
        dhdt = xr.polyval(years, derivative(data["coefs"], level=1))
        # print(dhdt)

        dhdt.sel(year=2017).plot(vmin=-3, vmax=3, cmap="RdBu")
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
            # plt.savefig(f"figures/dhdt/dhdt_{year_str}.jpg", dpi=300)
            plt.show()
        print(dhdt)
        return

        plt.show()


if __name__ == "__main__":
    stack_dems()
