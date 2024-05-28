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
import contextlib

import shapely
import sympy
import sympy.abc

import adsvalbard.rasters
import adsvalbard.utilities

import main


def make_dem_stack_mem(chunk_bounds: rio.coords.BoundingBox, res: list[float]):
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

    return data
    

def make_dem_stack(stack_filepath: Path, chunk_bounds: rio.coords.BoundingBox, res: list[float]) -> Path:
    if stack_filepath.is_file():
        return stack_filepath

    data =make_dem_stack_mem(chunk_bounds=chunk_bounds, res=res)

    data.to_netcdf(stack_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in data.data_vars})

    return stack_filepath


def interp(data: xr.Dataset, random_state: int = 0, degree=2, slope_corr: str = "per-timestamp"):
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


    data["diff_res"] = data["diff"] - data["diff"].median("time")

    res = data["diff_res"].to_dataframe().reset_index()

    # slope_corr = "No"

    if slope_corr == "per-timestamp":
        trend_y = data["diff"].polyfit("y", deg=1)["polyfit_coefficients"].isel(degree=0) * data["y"]
        data["diff"] -= trend_y
        data["diff"] -= data["diff"].polyfit("x", deg=1)["polyfit_coefficients"].isel(degree=0) * data["x"]
    elif slope_corr == "inter-timestamp":
        median_diff = data["diff"].median().item()
        model = sklearn.linear_model.LinearRegression().fit(res[["y", "x"]].values, res["diff_res"].values)
        yy, xx = xr.broadcast(data["y"], data["x"])
        # print(data["y"].broadcast_like(data["npi_dem"], ))
        data["corr"] = ("y", "x"), model.predict(np.transpose([yy.values.ravel(), xx.values.ravel()])).reshape((data["y"].shape[0], data["x"].shape[0]))
        data["diff"] += data["corr"]

        data["diff"] -= data["diff"].median().item() - median_diff


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
    if False:
        plt.title(f"{slope_corr.capitalize()} slope correction (R² = {score:.2f}, NMAD: {out['nmad'].item():.2f} m)")
        plt.scatter(data["time"] + data["npi_dem_year"].item(), data["diff"])
        xs = np.linspace(data["time"].min().item(), data["time"].max().item(), num=100)
        plt.plot(xs + data["npi_dem_year"].item(), model.predict(xs[:, None]), color="black")
        plt.ylim(-7, 1)
        plt.xlabel("Years")
        plt.ylabel("dH since start (m)")
        plt.show()
        raise NotImplementedError()
    return out

def determine_polynomials(out_filepath: Path, in_filepath: Path, degree: int = 2, slope_corr: str = "per-timestamp"):

    if out_filepath.is_file():
        return out_filepath
    with xr.open_dataset(in_filepath) as data:
        # data = data.isel(x=slice(200, 300), y=slice(200, 300))

        # data = data.sel(x=slice(551000, 551600), y=slice(8645377, 8644875))

      
        # data["npi_dem"].plot()
        # plt.show()
        # return

        # point = data.sel(x=poi.x, y=poi.y, method="nearest")

        # plt.scatter(point["time"], point["arcticdem"])
        # plt.show()

        coarsened = (
            data.coarsen(x=10, y=10)
            .construct(x=["x_coarse", "x"], y=["y_coarse", "y"])
            .stack(coarse=["y_coarse", "x_coarse"])
            .groupby("coarse")
            .map(interp, degree=degree, slope_corr=slope_corr)
            .unstack()
        )
        coarsened["y"] = coarsened["y"].isel(x_coarse=0)
        coarsened["x"] = coarsened["x"].isel(y_coarse=0)
        coarsened = coarsened.swap_dims(y_coarse="y", x_coarse="x").drop(["y_coarse", "x_coarse"])

        temp_path = out_filepath.with_suffix(out_filepath.suffix + ".tmp")
        coarsened.to_netcdf(temp_path)

        shutil.move(temp_path,out_filepath)

    return out_filepath

    
def get_linear_trend(point: xr.Dataset):

    point = point.dropna("year")

    xvals = point["year"].values.ravel()[:, None]
    yvals = point["arcticdem"].values.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sklearn.exceptions.UndefinedMetricWarning)
        model =sklearn.linear_model.RANSACRegressor(max_trials=10).fit(xvals, yvals)

    out = point[["xy", "y", "x"]]

    out["slope"] = "xy", model.estimator_.coef_ 
    out["intercept"] = "xy", [model.estimator_.intercept_]
    out["score"] = "xy", [model.score(xvals[model.inlier_mask_], yvals[model.inlier_mask_])]
    out["n_inliers"] = "xy", [np.count_nonzero(model.inlier_mask_)]
    out["n_values"] = "xy", [model.inlier_mask_.size]

    return out

def derivative(coefficients: xr.DataArray, coord_name: str = "degree") -> xr.DataArray:

    out = coefficients.copy()

    degrees = sorted(out[coord_name].values)
    for i, degree in enumerate(degrees):

        if (len(degrees) - 1) == i:
            out.loc[{coord_name: degree}] = 0.
        else:
            out.loc[{coord_name: degree}] = out.sel(**{"degree": degree + 1}) * (degree + 1)

    return out

    

def test_derivative():

    vals = xr.DataArray([[5, 4, 3, 2]], coords=[("x", [0.]), ("degree", [3, 2, 1, 0])])

    deriv = derivative(vals)

    assert np.array_equal(deriv.values.ravel(), np.array([0, 15, 8, 3])) 

    

def determine_trends(out_filepath: Path, in_filepath: Path, in_data: xr.Dataset | None = None, min_count_per_year: int = 3):

    intervals = [(2012., 2017.), (2015., 2020.), (2017., 2023.)]

    if out_filepath.is_file():
        return out_filepath

    with contextlib.ExitStack() as stack:
        if in_data is not None:
            data = in_data
        else:
            data = stack.enter_context(xr.open_dataset(in_filepath, chunks="auto"))
    # with xr.open_dataset(in_filepath, chunks="auto") as data:

        # data["year_d"] = (
        #     data["time"].dt.year
        #     + data["time"].dt.month / 12
        #     + data["time"].dt.day / (30 * 12)
        #     + np.linspace(0, 1e-9, data["time"].shape[0])
        # )
        # data = data.swap_dims(time="year_d")

        coarse = data.coarsen(x=10, y=10)
        data = coarse.median()
        # data["arcticdem_std"] = coarse.std()["arcticdem"]

        # print(data)

        


        # return
        grouped = data["arcticdem"].groupby("time.year")
        data["yearly_med"] = grouped.median()
        data["yearly_std"] = grouped.std()
        data["yearly_count"] = grouped.count()

        def get_minmax_year_range(values, years: np.ndarray):
            finites = np.isfinite(values)

            if da.count_nonzero(values) < 2:
                return [[[0, 0]]]
            min_idx = np.argmax(finites, axis=0)

            max_idx = values.shape[0] - np.argmax(finites[::-1], axis=0) - 1

            # out = da.zeros(shape=(2,) + min_idx.shape, chunks=((2,),) + min_idx.chunks)
            # out[0, :, :] = years[min_idx]
            # out[1, :, :] = years[max_idx]


            # return out
            return [years[min_idx], years[max_idx]]


        minmax = xr.apply_ufunc(get_minmax_year_range, data["yearly_med"].where(data["yearly_count"] >= 2), input_core_dims=[["year", "y", "x"]], output_core_dims=[["minmax", "y", "x"]], dask="allowed", kwargs={"years": data["year"].values}, dask_gufunc_kwargs={"output_sizes": {"minmax": 2}}, output_dtypes=[data["year"].values.dtype])

        data["min_poly_year"] = minmax.isel(minmax=0).clip(min=2013) 
        data["max_poly_year"] = minmax.isel(minmax=1)
        
        # data = data.sel(year=slice(2013, 2030))

        # data = data.where(data["yearly_count"] > 2)
        data["use_for_poly"] = (data["year"].broadcast_like(data["yearly_med"]) >= 2013) & (data["yearly_count"] >= 2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.RankWarning)
            res = data["yearly_med"].where(data["use_for_poly"]).polyfit("year", deg=3, full=True)

        # data["pred"] = xr.polyval(data["year"], res["polyfit_coefficients"])
        data["polyfit_coefficients"] = res["polyfit_coefficients"]

        temp_filepath = out_filepath.with_name(out_filepath.name + ".tmp")

        vars = ["min_poly_year", "max_poly_year", "yearly_med", "yearly_std", "yearly_count", "polyfit_coefficients"]
        data[vars].to_netcdf(temp_filepath, encoding={v: {"zlib": True, "complevel": 5} for v in vars})

        shutil.move(temp_filepath, out_filepath)
        return out_filepath
        # data["dhdt"] = xr.polyval(data["year"], derivative(data["polyfit_coefficients"]))

        for year in data["year"].values:
            data["dhdt"].sel(year=year).plot(vmin=-20, vmax=20, cmap="RdBu")
            plt.show()
        return

        data["residuals"] = data["pred"] - data["yearly_med"]

        data["nmad"] = 1.4826 * (np.abs(data["residuals"].mean("year") - data["residuals"])).median("year")

        # data["nmad"].plot(vmin=0, vmax=10)
        ((data["pred"].sel(year=2022) - data["pred"].sel(year=2013)) / 10).plot(vmin=-10, vmax=10, cmap="RdBu")
        plt.show()
        return

        points = dict(
            bad = dict(x=549720, y=8643237),
            bad2 = dict(x=549051, y=8641784),
            good = dict(x=549943, y=8643371),
            maybe_nonlinear = dict(x=549297, y=8644026)
        )

        for i, (name, coord) in enumerate(points.items()):
            
            plt.subplot(1, len(points), i + 1)

            point = data.sel(coord, method="nearest").compute()
            plt.title(name)
            plt.errorbar(point["year"], point["yearly_med"], yerr=point["yearly_std"])
            plt.plot(point["year"], point["pred"])
            for year in point["year"].values:
                row = point.sel(year=year)
                plt.annotate(row["yearly_count"].item(), (year, row["yearly_med"].item()))
        plt.show()

        return

        

        data["nmad"] = 1.4826 * (np.abs(data["yearly_med"].median("year") - data["yearly_med"])).median("year")

        data["stable_mask"] = data["nmad"] <= 2.

        fit = data["yearly_med"].where(~data["stable_mask"]).polyfit("year", deg=1, full=True)
        fit["rmse"] = np.sqrt(fit["polyfit_residuals"] / data["year"].shape[0])
        fit["poly_degree"] = ("y", "x"), da.ones_like(fit["rmse"], dtype=int)
        fit["mean_slope"] = fit["polyfit_coefficients"].sel(degree=1)
        fit["nonlinear_mask"] = (fit["rmse"] / data["nmad"]) > 2

        for col in ["polyfit_residuals", "rmse", "poly_degree", "mean_slope"]:
            fit[col] = fit[col].where(~data["stable_mask"], other=0.)

        # fit["mean_slope"].where(~fit["nonlinear_mask"]).plot(vmin=-5, vmax=5)
        # plt.show()
        # return
        fit2 = data["yearly_med"].where(fit["nonlinear_mask"]).polyfit("year", deg=2, full=True)

        fit2["rmse"] = np.sqrt(fit2["polyfit_residuals"] / data["year"].shape[0])
        fit2["poly_degree"] = ("y", "x"), da.ones_like(fit2["rmse"], dtype=int)
        fit2["mean_slope"] = 2 * (fit2["polyfit_coefficients"].sel(degree=2) * (data["year"].isel(year=0).item() + data["year"].isel(year=-1).item()) + fit2["polyfit_coefficients"].sel(degree=1)) / (data["year"].isel(year=0).item() - data["year"].isel(year=-1).item()) 

        for col in ["polyfit_residuals", "rmse", "poly_degree", "mean_slope"]:
            fit2[col] = fit2[col].where(fit["nonlinear_mask"], other=fit[col])

        linear_coef = fit["polyfit_coefficients"].broadcast_like(fit2["polyfit_coefficients"])
        linear_coef.loc[{"degree": 2}] = 0.
        fit2["polyfit_coefficients"] = fit2["polyfit_coefficients"].where(fit["nonlinear_mask"], other=linear_coef.sel(degree=fit2["degree"].values))

        fit2["robust_fit_mask"] = (fit2["rmse"] / data["nmad"]) > 2.5
        # plt.subplot(1, 2, 1)
        # fit2["rmse"].plot(vmin=0, vmax=20)
        # plt.subplot(1, 2, 2)
        # fit2["robust_fit_mask"].plot()
        # plt.show()
        # return

        # fit2["mean_slope"].plot(vmin=-1, vmax=1)

        # fit2["rmse"].plot(vmin=0, vmax=20)
        # plt.show()
        # return

        robust = xr.apply_ufunc(trend, data["yearly_med"].where(fit2["robust_fit_mask"]).chunk(year=-1), input_core_dims=[["year"]], kwargs={"times": data["year"].values}, output_core_dims=[["coef"]], dask="parallelized", dask_gufunc_kwargs={"output_sizes": {"coef": 6}}).to_dataset(name="res")


        robust["polyfit_coefficients"] = robust.rename_dims(coef="degree")["res"].isel(degree=slice(0, 2))
        robust["degree"] = "degree", [1, 0]

        robust["pred"] = xr.polyval(data["year"], robust["polyfit_coefficients"])
        robust["residuals"] = data["yearly_med"] - robust["pred"]
        robust["rmse"] = (robust["residuals"] ** 2).mean("year") ** 0.5
        robust["mean_slope"] = robust["polyfit_coefficients"].sel(degree=1)
        robust["poly_degree"] = ("y", "x"), da.ones_like(robust["rmse"], dtype=int)


        for col in ["rmse", "poly_degree", "mean_slope"]:
            fit2[col] = fit2[col].where(~fit2["robust_fit_mask"], other=robust[col])

        robust_coef = robust["polyfit_coefficients"].broadcast_like(fit2["polyfit_coefficients"])
        robust_coef.loc[{"degree": 2}] = 0

        fit2["polyfit_coefficients"] = fit2["polyfit_coefficients"].where(~fit2["robust_fit_mask"], other=robust_coef.sel(degree=fit2["degree"].values))
        

       

        with tqdm.dask.TqdmCallback():
            fit2.compute()

        
       
        # robust = fit2["arcticdem"].where(fit2["robust_fit_mask"]).reduce(

        data["model"] = xr.polyval(data["year"], fit2["polyfit_coefficients"])
        bad = dict(x=549720, y=8643237)
        good = dict(x=549943, y=8643371)
        maybe_nonlinear = dict(x=549297, y=8644026)


        fit2["mean_slope"].plot()
        plt.show()
        # plt.subplot(1, 3, 1)
        # data["yearly_med"].sel(**bad, method="nearest").plot()
        # data["model"].sel(**bad, method="nearest").plot()
        # plt.subplot(1, 3, 2)
        # data["yearly_med"].sel(**good, method="nearest").plot()
        # data["model"].sel(**good, method="nearest").plot()
        # plt.subplot(1, 3, 3)
        # data["yearly_med"].sel(**maybe_nonlinear, method="nearest").plot()
        # data["model"].sel(**maybe_nonlinear, method="nearest").plot()

        # # fit2["rmse"].plot(vmin=0, vmax=20)
        # plt.show()

        # return

        # fit2["polyfit_coefficients"] = fit2["polyfit_coefficients"].where(~fit["nonlinear_mask"], other=0.)
        # fit2["polyfit_coefficients"].where(~fit["nonlinear_mask"]).loc[{"degree": slice(0, 2)}] = fit["polyfit_coefficients"]
            # fit2[col]

        print(fit2)
        return

        # fit2["polyfit_coefficients"].sel(degree=1).plot()

        # fit2.where(fit["nonlinear_mask"]

        fit["polyfit_coefficients"].where(~fit["nonlinear_mask"]).sel(degree=1).plot()
        
        plt.show()

        
        return

        plt.subplot(1, 2, 1)

        plt.subplot(1, 2, 2)
        fit["polyfit_coefficients"].sel(degree=1).plot()
        plt.show()

        print(fit)

        # data["yearly_med"].sel(year=2021).plot()

        # (data["yearly_med"].sel(year=2021) - data["yearly_med"].sel(year=2013)).plot()
        # data["yearly_med"].sel(x=550030, y=8644360, method="nearest").plot()
        # plt.show()

        # print(data)
        return
        final = []
        for interval in intervals:
            subset = data.sel(year=slice(*interval))

            subset = subset.coarsen(x=10, y=10).median()#.stack(xy=["y", "x"])


            # subset = subset.isel(x=slice(0, 50), y=slice(0, 5))

            # res = xr.apply_ufunc(trend, subset["arcticdem"].chunk(year=-1), input_core_dims=[["year"]], kwargs={"times": subset["year"].values}, output_core_dims=[["coef"]], dask="parallelized", dask_gufunc_kwargs={"output_sizes": {"coef": 6}})

            # out = xr.Dataset()
            # for i, name in enumerate(["slope", "intercept", "n_inliers", "n_total", "score", "nmad"]):
            #     if "n_" in name:
            #         out[name] = res.isel(coef=i).astype(int)
            #     else:
            #         out[name] = res.isel(coef=i)
            out = subset[["arcticdem"]]

            return

            out = out.expand_dims({"interval": [np.mean(interval)]}, axis=0)
            out["interval_min"] = "interval", [interval[0]]
            out["interval_max"] = "interval", [interval[1]]

            final.append(out)

        final = xr.combine_nested(final, "interval")


        final["med"] = final["arcticdem"].median(["year"])

        final["med_diff"] = final["med"].diff("interval") / final["interval"].diff("interval")

        plt.subplot(1, 2, 1)
        final["med_diff"].isel(interval=-2).plot()
        plt.subplot(1, 2, 2)
        final["med_diff"].isel(interval=-1).plot()
        plt.show()


        print(final)
        return
        robust = final["med"].polyfit("interval", deg=1)


        print(robust)

        robust["polyfit_coefficients"].sel(degree=1).plot()
        plt.show()

        print(robust)
        return
        task = final.to_netcdf("./test.nc.temp", compute=False)

        with tqdm.dask.TqdmCallback():
            task.compute(scheduler="threads")
        print(final)

def trend(arr: da.array, times: np.ndarray):
    shape = arr.shape[:2]

    arr = arr.reshape((-1, arr.shape[2]))

    res = da.apply_along_axis(trend_inner, axis=1, arr=arr, times=times)

    return res.reshape(shape + res.shape[1:])
    # return np.mean(arr, axis=)

def trend_inner(elevation: np.ndarray, times: np.ndarray):
    valid_mask = da.isfinite(elevation)

    if da.count_nonzero(valid_mask) < 3:
        return np.zeros((6), dtype="float32") + np.nan

    elevation =elevation[valid_mask]
    times = times[valid_mask, None]

    model = sklearn.linear_model.RANSACRegressor(random_state=0).fit(times, elevation)

    res = elevation - model.predict(times)

    score = model.score(times, elevation)

    retval = np.array(
        [
            model.estimator_.coef_[0],
            model.estimator_.intercept_,
            np.count_nonzero(model.inlier_mask_),
            model.inlier_mask_.size,
            1.4826 * np.median(np.abs(res - np.median(res))),
            score,
        ]
    )

    return retval

           
def stack_dems(stack_res: float = 50.0, chunk_size: int = 1000):
    bounds = main.get_bounds(region="heerland")
    res = main.get_res()

    print("Starting")
    out_bounds = adsvalbard.utilities.align_bounds(bounds, [stack_res] * 2)
    chunks = adsvalbard.rasters.generate_raster_chunks(out_bounds, res=res[0], chunksize=1000)

    filepaths = []
    for i, chunk_bounds in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
        chunk_bnd_box = shapely.geometry.box(*chunk_bounds)

        poi = shapely.geometry.Point(550030, 8644360)

        # if not chunk_bnd_box.contains(poi):
        #     continue

        filepath = Path(f"temp/dem_stack_chunk_{str(i).zfill(3)}.nc")
        coef_filepath = filepath.with_stem(filepath.stem.replace("dem_stack", "coeffs"))

        # print("Making DEM stack")
        # make_dem_stack(filepath, chunk_bounds=chunk_bounds, res=res)

        # determine_trends(filepath)
        if not coef_filepath.is_file():
            data =make_dem_stack_mem(chunk_bounds=chunk_bounds, res=res)
            determine_trends(coef_filepath, None, data)

        filepaths.append(coef_filepath)

        continue

        print("Determining polynomials")

        for corr in ["no", "per-timestamp", "inter-timestamp"]:

            corr_filepath =coef_filepath.with_stem(coef_filepath.stem + (f"_{corr}" if corr != "no" else ""))

            print(corr_filepath)
            
            determine_polynomials(corr_filepath, filepath, slope_corr=corr)
            with xr.open_dataset(corr_filepath) as data:

                print(data["score"].mean().item())
                # continue

                plt.figure(figsize=(17, 4))
                plt.suptitle(f"{corr.capitalize()} slope corr.")
                cbar_args = dict(fraction=0.03, aspect=20)
                plt.subplot(1, 4, 1)
                extent = [data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()]
                plt.imshow(data["coefs"].sel(degree=2) * 2 * 2022 + data["coefs"].sel(degree=1), cmap="RdBu", vmin=-8, vmax=8, extent=extent)
                cbar = plt.colorbar(**cbar_args)
                cbar.set_label("Rate (m / a)")

                plt.subplot(1, 4, 2)
                extent = [data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()]
                plt.imshow(data["coefs"].sel(degree=2) * 2, cmap="PRGn", vmin=-2, vmax=2, extent=extent)
                cbar = plt.colorbar(**cbar_args)
                cbar.set_label("Acceleration (m / a²)")

                
                plt.subplot(1, 4, 3)
                plt.imshow(data["score"], cmap="Greys_r", vmin=0, vmax=1, extent=extent)
                cbar = plt.colorbar(**cbar_args)
                cbar.set_label("R² score")
                plt.xticks([])
                plt.yticks([])

                plt.subplot(1, 4, 4)
                plt.imshow(data["nmad"], cmap="Reds", vmin=0, vmax=5, extent=extent)
                cbar = plt.colorbar(**cbar_args)
                cbar.set_label("NMAD (m)")
                plt.xticks([])
                plt.yticks([])

                plt.tight_layout()

                Path("figures/").mkdir(exist_ok=True)
                plt.savefig(f"figures/poly_{corr}_corr.jpg", dpi=300)
                # data["coefs"].sel(degree=1).plot()
                # data["diff"] = data["elev1"] - data["elev0"]
                # data["coefs"].sel(degree=2).plot()
                plt.close()
                # return

        corr_filepath =coef_filepath.with_stem(coef_filepath.stem + ("_per-timestamp"))

    with xr.open_mfdataset(filepaths, chunks="auto") as data:

        years = np.linspace(2013, 2022)
        years = xr.DataArray(years, coords=[("year", years)] )
        data["polyfit_coefficients"] = data["polyfit_coefficients"].isel(year=0)


        def get_minmax_year_range(values, years: np.ndarray):
            finites = np.isfinite(values)

            if da.count_nonzero(values) < 2:
                return [[[0, 0]]]
            min_idx = np.argmax(finites, axis=0)

            max_idx = values.shape[0] - np.argmax(finites[::-1], axis=0) - 1

            out = da.zeros(shape=(2,) + min_idx.shape, chunks=((2,),) + min_idx.chunks)
            out[0, :, :] = years[min_idx]
            out[1, :, :] = years[max_idx]


            return out
            # return [years[min_idx], years[max_idx]]


        minmax = xr.apply_ufunc(get_minmax_year_range, data["yearly_med"], input_core_dims=[["year", "y", "x"]], output_core_dims=[["minmax", "y", "x"]], dask="allowed", kwargs={"years": data["year"].values}, dask_gufunc_kwargs={"output_sizes": {"minmax": 2}}, output_dtypes=[data["year"].values.dtype])

        point = data.sel(x=[564212], y=[8623608], method="nearest").compute()

        point["min_year"] = minmax.isel(minmax=0)
        point["max_year"] = minmax.isel(minmax=1)

        print(point)
        point = point.isel(x=0, y=0).compute()

        print(point)
        point["yearly_med"].plot()
        xr.polyval(point["year"], point["polyfit_coefficients"]).plot()
        ylim = plt.gca().get_ylim()
        plt.vlines([point["min_year"], point["max_year"]], *ylim)
        plt.ylim(ylim)
        plt.show()

        print(point)
        return
        data["yearly_med"].sel(year=2021).plot()
        plt.show()
        print(data)
        return

        dhdt = xr.polyval(years.coords["year"], derivative(data["polyfit_coefficients"]))

        for year in dhdt["year"].values:
            plt.figure(figsize=(7, 5))
            dhdt.sel(year=year).plot(vmin=-10, vmax=10, cmap="RdBu")
            plt.show()
        print(dhdt)
        return

        plt.show()


if __name__ == "__main__":
    stack_dems()
