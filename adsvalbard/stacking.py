import json
import shutil
import warnings
from pathlib import Path
import threading
import functools
import shutil

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import tqdm
import tqdm.dask
import tqdm.contrib.concurrent
import xarray as xr
import zarr
from dask.array.core import PerformanceWarning

import rasterio
import rasterio.coords
import rasterio.transform
import rasterio.windows

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS


def create_stack(region: str = "heerland") -> Path:
    # TODO: Handle nodata correctly! I had lots of -9999. in the ArcticDEM stack series.

    # dem_paths = list(Path("temp.svalbard/heerland_dem_coreg/").glob("*/*dem_coreg.tif"))

    out_path = Path(f"temp.svalbard/stacks/dem_stack_{region}.zarr")
    temp_path = out_path.with_suffix(".zarr.tmp")

    if temp_path.is_dir():
        shutil.rmtree(temp_path)

    if out_path.is_dir():
        return out_path

    import rasterio.coords
    import rioxarray

    mask_stems = list(
        map(
            lambda fp: fp.stem.replace("_outlier_proba", ""),
            Path("temp.svalbard/outlier_proba/").glob("*/*.tif"),
        )
    )

    dem_paths = list(
        filter(
            lambda fp: fp.stem in mask_stems,
            Path(f"temp.svalbard/{region}_dem_coreg/").glob("*/*dem_coreg.tif"),
        )
    )

    bounds = CONSTANTS.regions[region]
    res = CONSTANTS.res
    shape = adsvalbard.utilities.get_shape(
        rasterio.coords.BoundingBox(**bounds), [res] * 2
    )

    xr_coords = {
        "y": np.linspace(bounds["bottom"] + res / 2, bounds["top"] - res / 2, shape[0])[
            ::-1
        ],
        "x": np.linspace(bounds["left"] + res / 2, bounds["right"] - res / 2, shape[1]),
    }

    dates = [
        (filepath, pd.to_datetime(filepath.stem.split("_")[3], format="%Y%m%d"))
        for filepath in dem_paths
    ]
    dates.sort(key=lambda tup: tup[1])
    # dems = xr.Dataset(coords=xr_coords)

    # print(dems)
    # return

    dems = []
    for filepath, date in tqdm.tqdm(dates):
        mask_path = (
            filepath.parent.parent.parent
            / f"outlier_proba/{filepath.parent.stem}/{filepath.stem}_outlier_proba.tif"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dem = rioxarray.open_rasterio(filepath, chunks="auto")
            mask = rioxarray.open_rasterio(mask_path, chunks="auto")

        mask = mask.isel(band=0).drop_vars(["band", "spatial_ref"])

        dem = (
            dem.isel(band=0)
            .sel(
                x=slice(bounds["left"], bounds["right"]),
                y=slice(bounds["top"], bounds["bottom"]),
            )
            .to_dataset(name="elevation")
            .drop_vars(["band", "spatial_ref"])
            .expand_dims(filename=[filepath.stem.replace("_dem_coreg", "")])
            .assign_coords(bounds=["xmin", "ymin", "xmax", "ymax"])
        )
        dem["elevation"].attrs = {}
        dem["bounding_box"] = (
            ("filename", "bounds"),
            [[dem["x"].min(), dem["y"].min(), dem["x"].max(), dem["y"].max()]],
        )

        with open(filepath.with_suffix(".json")) as infile:
            meta = json.load(infile)

        dem["stable_terrain_nmad"] = ("filename",), [meta["stable_nmad"]]
        dem["stable_terrain_frac"] = ("filename",), [meta["stable_fraction"]]
        dem["approx_vshift"] = (
            ("filename",),
            [meta["steps"][0]["meta"]["_meta"]["matrix"][-2][-1]],
        )
        dem["date"] = ("filename",), [date]

        dem["outlier_proba"] = mask

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            dems.append(
                dem.reindex(
                    xr_coords, fill_value={"outlier_proba": np.uint8(255)}
                ).chunk(x=512, y=512)
            )

        # if len(dems) > 10:
        #     break

    dems = xr.combine_nested(dems, "filename")
    dems["filename"] = dems["filename"].astype(str)

    for key in map(str, dems.data_vars):
        if key in ["elevation", "outlier_proba"]:
            continue
        dems[key] = dems[key].chunk(filename=-1)

    for name in ["npi_mosaic.vrt", "npi_mosaic_years.vrt", "stable_terrain.tif"]:
        filepath = CONSTANTS.temp_dir / name
        key = filepath.stem.replace("_mosaic", "_dem")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = (
                rioxarray.open_rasterio(filepath, chunks="auto")
                .isel(band=0)
                .drop_vars(["band", "spatial_ref"])
            )

        dems[key] = data.sel(**xr_coords).chunk(x=512, y=512)

    task = dems.to_zarr(
        temp_path,
        encoding={
            key: {"compressor": zarr.Blosc(cname="zstd", clevel=5, shuffle=2)}
            for key in ["elevation", "outlier_proba"]
        },
        compute=False,
    )

    with tqdm.dask.TqdmCallback(desc="Saving stack", smoothing=0.1):
        task.compute()

    shutil.move(temp_path, out_path)

    return out_path


def main():
    subsets = {
        "vallakra": {"xmin": 547000, "xmax": 554000, "ymin": 8640000, "ymax": 8648000},
        "tinkarp": {
            "xmin": 548762.5,
            "ymin": 8656585,
            "xmax": 553272.5,
            "ymax": 8658480,
        },
    }

    stack_path = create_stack()

    glaciers = gpd.read_file("shapes/glacier_outlines.sqlite")

    glacier_name = "N Fredbreen"
    # glacier_name = "Tinkarpbreen"

    points = {
        "N Fredbreen": {
            "advancing_front": {"x": 545170, "y": 8626160},
            "scheele": {
                "x": 545783,
                "y": 8626293,
                "desc": "Surge in 2021",
                "plot": True,
            },
            "trend0": {"x": 543317.5, "y": 8627532.5},
            "trend1": {
                "x": 543312.5,
                "y": 8627532.5,
                "desc": "Slow surge",
                "plot": True,
            },
        },
        "Morsnevbreen": {
            "tidewater_front": {
                "x": 563500,
                "y": 8612030,
                "desc": "Tidewater front advance",
                "plot": True,
            },
            "accumulation_area": {
                "x": 567018,
                "y": 8625060,
                "desc": "Acc. area during/post surge",
                "plot": True,
                "max_z": 500,
            },
        },
        "Scheelebreen": {
            "coast_point": {
                "x": 549063,
                "y": 8633541,
                "desc": "Near the coast",
                "plot": True,
                "max_z": 500,
            },
            "accumulation_area": {
                "x": 547717,
                "y": 8622038,
                "desc": "Accumulation area surge",
                "plot": True,
            },
        },
        "Vallåkrabreen": {
            "surge_bulge": {
                "x": 549520,
                "y": 8643117,
                "desc": "Surge bulge evolution",
                "plot": True,
                "max_z": 500,
            },
        },
        "Arnesenbreen": {
            "accumulation_area": {
                "x": 572236,
                "y": 8635558,
                "desc": "Accumulation area surge",
                "plot": True,
            },
        },
        "Tinkarpbreen": {
            "accumulation_area": {
                "x": 550486,
                "y": 8656975,
                "desc": "Accumulation area draining",
                "plot": True,
            }
        },
        "Klubbebreen": {
            "slow_surge_front": {
                "x": 548960,
                "y": 8628312,
                "desc": "Slow-surge front",
                "plot": True,
            },
        },
        "Edvardbreen": {
            "surge_bulge": {
                "x": 560374,
                "y": 8646527,
                "desc": "Surge bulge front",
                "plot": True,
            },
        },
        "Kvalbreen": {
            "retreat_then_surge": {
                "x": 569428,
                "y": 8610473,
                "desc": "Terminus retreat, then surge",
                "plot": True,
            },
        },
    }

    # points_to_plot = {"N Fredbreen": ["scheele", "trend1"], "Morsnevbreen": ["tidewater_front"], "Scheelebreen": ["coast_point"], "Vallåkrabreen": ["surge_bulge"], "Arnesenbreen": ["accumulation_area"]}

    points_to_plot = {
        glac: {key: point for key, point in pts.items() if point.get("plot", False)}
        for glac, pts in points.items()
    }

    with xr.open_zarr(stack_path) as data:
        data["elevation"] = data["elevation"].where(data["elevation"] != -9999.0)

        data["outlier_proba"] = data["outlier_proba"].astype("float32") / 255

        n_panels = sum(len(point_dict.keys()) for point_dict in points_to_plot.values())
        n_rows = int(np.sqrt(n_panels))
        n_cols = int(np.ceil(n_panels / n_rows))

        print(n_panels, n_rows, n_cols)

        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
        axes = fig.subplots(n_rows, n_cols, sharex=True)
        panel_i = 0
        for glacier_name in points_to_plot:
            # glacier = glaciers.query(f"glac_name == '{glacier_name}'")
            # bounds = dict(zip(["xmin", "ymin", "xmax", "ymax"], glacier.geometry.buffer(500).total_bounds))
            # subset = data.sel(x=slice(bounds["xmin"], bounds["xmax"]), y=slice(bounds["ymax"], bounds["ymin"]))
            # mask = (
            #     (subset["bounding_box"].sel(bounds="xmin") < data["x"].max()) &
            #     (subset["bounding_box"].sel(bounds="xmax") > data["x"].min()) &
            #     (subset["bounding_box"].sel(bounds="ymin") < data["y"].max()) &
            #     (subset["bounding_box"].sel(bounds="ymax") > data["y"].min())
            # )
            # overlapping = mask["filename"].where(mask.compute(), drop=True)
            # subset = subset.sel(filename=overlapping)
            subset = data

            for point_key in points_to_plot[glacier_name]:
                axis: plt.Axes = axes.ravel()[panel_i]
                point_coord = points_to_plot[glacier_name][point_key]
                point = subset.sel(
                    x=point_coord["x"], y=point_coord["y"], method="nearest"
                ).compute().dropna("filename", subset=["elevation"])

                extreme_outliers = (
                    (point["outlier_proba"] > 0.95) | (~np.isfinite(point["elevation"])) | (point["elevation"] > points_to_plot[glacier_name][point_key].get("max_z", 1000))
                )  # | (np.abs(point["elevation"] - point["elevation"].median("filename")) > 300)

                point["elevation"] = point["elevation"].where(~extreme_outliers)
                axis.scatter(
                    point["date"],
                    point["elevation"],
                    c=point["outlier_proba"] * 100,
                    zorder=2,
                    vmin=50,
                    vmax=100,
                )
                ylim = axis.get_ylim()

                axis.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.1)
                ylim = axis.get_ylim()

                yticks = np.arange(
                    ylim[0] - ylim[0] % 10, ylim[1] + 10, step=10
                ).astype(int)
                if yticks.shape[0] > 5:
                    axis.set_yticks(yticks, minor=True)
                    # print(list(filter(lambda i, tick: i % 3 == 0, enumerate(yticks)))
                    # axis.set_yticks(list(map(lambda i, tick: tick, filter(lambda i, tick: i % 3 == 0, enumerate(yticks)))))
                    axis.set_yticks(yticks[::3])
                    # axis.set_yticklabels(["" if i % 3 != 0 else str(tick) for i, tick in enumerate(yticks)])
                else:
                    axis.set_yticks(yticks)

                axis.text(
                    0.01,
                    0.99,
                    s="abcdefghijklmnopq"[panel_i] + ")",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                )

                if "desc" in point_coord:
                    axis.text(
                        0.5,
                        0.99,
                        s=f"{glacier_name}\n{point_coord['desc']}",
                        transform=axis.transAxes,
                        ha="center",
                        va="top",
                    )

                if (n_outliers := np.count_nonzero(extreme_outliers.values)) > 0:
                    axis.text(
                        0.99,
                        0.01,
                        s=f"Hidden {n_outliers} extreme outlier(s)",
                        transform=axis.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=7,
                    )

                axis.grid(which="both", color="lightgray", linestyle="--", zorder=1)

                panel_i += 1

        plt.text(
            0.01,
            0.5,
            "Elevation (m a.s.l.)",
            transform=fig.transFigure,
            rotation=90,
            va="center",
            ha="left",
        )
        plt.subplots_adjust(
            left=0.052, bottom=0.036, right=0.995, top=0.974, wspace=0.188, hspace=0.155
        )
        plt.savefig("figures/elevation_series_examples.pdf")
        # plt.tight_layout()
        plt.show()

        return

        # for i, (glacier_name, point_key) in enumerate(points_to_plot):

        poi_key = "scheele"
        point = data.sel(
            x=[points[glacier_name][poi_key]["x"]],
            y=[points[glacier_name][poi_key]["y"]],
            method="nearest",
        ).isel(x=0, y=0)

        plt.figure(dpi=200)
        plt.scatter(point["date"], point["elevation"], c=point["outlier_proba"] * 100)
        cbar = plt.colorbar()
        cbar.set_label("Outlier probability (%)")
        plt.ylabel("Elevation (m a.s.l.)")
        plt.tight_layout()

        plt.show()
        return

        print(point)

        data = data.sel(
            x=slice(bounds["xmin"], bounds["xmax"]),
            y=slice(bounds["ymax"], bounds["ymin"]),
        )

        mask = (
            (data["bounding_box"].sel(bounds="xmin") < data["x"].max())
            & (data["bounding_box"].sel(bounds="xmax") > data["x"].min())
            & (data["bounding_box"].sel(bounds="ymin") < data["y"].max())
            & (data["bounding_box"].sel(bounds="ymax") > data["y"].min())
        )
        overlapping = mask["filename"].where(mask.compute(), drop=True)
        data = data.sel(filename=overlapping)

        # data["weight"] = 1 / (1 - data["outlier_proba"] ** 2)

        return

        # data["yearly_med_filt"] = data["elevation"].where(data["outlier_proba"] < 0.8).groupby(data["date"].dt.year).median()

        # data["elevation_count"] = data["elevation"].groupby(data["date"].dt.year).count()
        # enough_data = (data["elevation"].groupby(data["date"].dt.year).count() > 1)

        # data["yearly_med_filt"] = data["yearly_med_filt"].where(enough_data)

        data[["elevation", "outlier_proba"]] = data[
            ["elevation", "outlier_proba"]
        ].where((data["outlier_proba"] < 0.6))
        data["elevation_count"] = (
            data["elevation"].groupby(data["date"].dt.year).count()
        )
        enough_data = data["elevation"].groupby(data["date"].dt.year).count() > 1
        # data[["elevation", "outlier_proba"]] = data[["elevation", "outlier_proba"]].where((enough_data.sel(year=data["date"].dt.year)))
        data["yearly_med"] = data["elevation"].groupby(data["date"].dt.year).median()

        # data = data.assign_coords(date_yr=data["date"].dt.year + data["date"].dt.month / 12 + data["date"].dt.day / (12 * 30))
        data["date_yr"] = (
            data["date"].dt.year
            + data["date"].dt.month / 12
            + data["date"].dt.day / (12 * 30)
        )
        data["yearly_med_date"] = (
            data["date_yr"].load().groupby(data["date"].dt.year).median()
        )

        # data["yearly_med"] = data["yearly_med"].chunk(year=-1).interpolate_na("year").compute()

        data["diff"] = data["yearly_med"].diff("year").compute()

        for panel_i, (year, yearly) in enumerate(
            data["diff"].groupby("year", squeeze=False)
        ):
            vals = yearly.values.squeeze()

            if np.count_nonzero(np.isfinite(vals)) == 0:
                continue

            if panel_i == 0:
                continue

            plt.title(f"{year} - {year - 1}")
            plt.imshow(vals, vmin=-5, vmax=5, cmap="RdBu")
            plt.show()

        # data["yearly_med"].sel(year=2021).plot()
        # plt.show()
        return

        # data["polyfit"] = data["elevation"].swap_dims(filename="date_yr").polyfit("date_yr", deg=6)["polyfit_coefficients"]

        # data["npi_dem"].plot()
        # plt.show()

        # x_coords = data["date_yr"].swap_dims(filename="date_yr").sortby("date_yr")

        # pred = xr.polyval(x_coords, data["polyfit"])

        # def interp(point):
        #     point = point.resample(date=pd.Timedelta(days=120)).map(lambda ds: ds.weighted(ds["weight"]).mean("date"))
        #     point = point.dropna(subset=["elevation"])

        #     print(point)

        # print(point.swap_dims(filename="date").stack(xy=["x", "y"]).groupby("xy"))

        # return

        data = data.isel(x=slice(50), y=slice(50))

        # means =( data[["elevation", "weight", "date", "outlier_proba"]]
        #     .swap_dims(filename="date")
        #     # .chunk(date=-1)
        #     .resample(date=pd.Timedelta(days=365))
        #     .map(lambda ds: ds.weighted(ds["weight"]).mean(["date"]))
        # )
        # means["date_yr"] = means["date"].dt.year + means["date"].dt.month / 12 + means["date"].dt.day / (12 * 30)

        # data["year_dt"] = "year", pd.to_datetime(data["year"].astype(str).astype(object) + "-01-01", format="%Y-%m-%d")
        # print(data.sel(**points[glacier_name]["trend0"])["elevation_count"].load())
        # print(data.sel(**points[glacier_name]["trend0"])["elevation"].load())
        # data.sel(**points[glacier_name]["trend0"]).plot.scatter(x="year_med_date", y="yearly_med", color="black")
        # data.sel(**points[glacier_name]["trend0"]).plot.scatter(x="date", y="elevation", hue="outlier_proba", s=50)
        # data.sel(**points[glacier_name]["trend1"]).plot.scatter(x="year_med_date", y="yearly_med", marker="s", color="black")
        # data.sel(**points[glacier_name]["trend1"]).plot.scatter(x="date", y="elevation", hue="outlier_proba",  marker="s", s=50)
        # plt.show()
        # return

        def interp_inner(vals: np.ndarray, times: np.ndarray):
            years = np.arange(2013, 2022)

            valid_mask = np.isfinite(vals)

            if np.count_nonzero(valid_mask) == 0:
                return np.zeros_like(years) + np.nan

            return scipy.interpolate.interp1d(times.ravel(), vals.ravel())(years)

        # def interp_func(ds: xr.Dataset):
        # years = np.arange(2013, 2022)

        # ds = ds.dropna("date_yr", subset=["elevation"])

        # res = xr.apply_ufunc(
        #     interp_inner,
        #     ds["elevation"],
        #     kwargs=dict(times=ds["date_yr"]),
        # )

        # return res

        # ds.coords["year"] = "year", years
        # ds["interp"] = "year", scipy.interpolate.interp1d(ds["date_yr"].values.ravel(), ds["elevation"].values.ravel())(years)

        # return ds["interp"]

        import dask.array as da

        def interp_inner(
            arr: da.Array,
            times: np.ndarray,
            orig_shape: tuple[int, ...],
            years: np.ndarray,
        ):
            valid_mask = da.isfinite(arr)

            if da.count_nonzero(valid_mask) < 2:
                return da.zeros_like(years) + np.nan

            model = scipy.interpolate.interp1d(
                times[valid_mask], arr[valid_mask], fill_value="extrapolate"
            )

            return model(years)

        def interp_func(arr: da.Array, times: np.ndarray, years: np.ndarray):
            res = da.apply_along_axis(
                interp_inner,
                axis=1,
                arr=arr.reshape((-1, arr.shape[-1])),
                orig_shape=arr.shape[:2],
                times=times,
                years=years,
            )
            res = res.reshape((arr.shape[:2] + (-1,)))

            return res

        # interp = means.swap_dims(date="date_yr").stack(xy=["x", "y"])

        # interp = means.swap_dims(date="date_yr").stack(xy=["x", "y"]).compute().groupby("xy", squeeze=False).map(interp_func).unstack()

        # subset_res = xr.apply_ufunc(
        #     get_poly,
        #     subset["d_h"],
        #     input_core_dims=[["coarse", "y", "x", "time"]],
        #     output_core_dims=[["coarse", "retvals", "order"]],
        #     output_dtypes=[np.float64],
        #     dask_gufunc_kwargs={
        #         "output_sizes": {"retvals": 7 + max(degree),"order": len(degree)},

        #     },
        #     kwargs={
        #         "res": 5.0,
        #         "times": coarsened["time"].values - year,
        #         "npi_dem_year": year,
        #         "degree": degree
        #     },
        #     dask="parallelized",

        # )

        years = np.arange(2012, 2024)
        interp = xr.apply_ufunc(
            interp_func,
            data["yearly_med"],
            input_core_dims=[["y", "x", "year"]],
            output_core_dims=[["y", "x", "year"]],
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "year": len(years),
                },
                "allow_rechunk": True,
            },
            kwargs={
                "times": data["year"].values,
                "years": years,
            },
            dask="allowed",
        )
        interp.coords["year"] = "year", years

        # interp.isel(x=0, y=0).plot()
        # plt.show()

        data["interp"] = interp

        with tqdm.dask.TqdmCallback():
            panel_i = data["interp"].compute().load()

            print(panel_i)

        panel_i.sel(year=2022).plot()
        plt.show()

        # out =

        return

        # print(point.swap_dims(filename="date").sortby("date").resample(date=pd.Timedelta(days=120)).map(lambda ds: ds.weighted(ds["weight"]).mean()))
        # return

        # point["elevation"].swap_dims(filename="date_yr").reset_coords(drop=True).plot.scatter()
        point.plot.scatter(x="date_yr", y="elevation", hue="outlier_proba")
        # print(point.swap_dims(filename="date").sortby("date").resample(date=pd.Timedelta(days=90)).mean())
        (
            point.swap_dims(filename="date")
            .resample(date=pd.Timedelta(days=120))
            .map(lambda ds: ds.weighted(ds["weight"]).mean())
            .plot.scatter(x="date_yr", y="elevation", color="black")
        )

        # pred.sel(x=point["x"], y=point["y"]).plot()
        plt.show()
        # print(point)
        return
        # data["yearly_med_filt"].isel(x=200, y=200).plot()
        # data["yearly_med_filt"].isel(year=-1).plot()
        plt.show()

        # data["npi_dem"].plot()
        # plt.show()

        # print(data)
        return

        # data = data.set_coords("date")
        # return
        data["polyfit"] = (
            data["elevation"]
            .swap_dims(filename="date_yr")
            .polyfit("date_yr", deg=1)["polyfit_coefficients"]
        )

        data["poly_nmad"] = xr.polyval(data["date_yr"], data["polyfit"]).median(
            "filename"
        )

        data["poly_nmad"].plot()
        plt.show()
        print(data)

        # plt.hist(data["stable_terrain_nmad"], bins=50)
        # plt.show()

        # print(data)
        #
        #
        #

def process_chunk(filepath: Path, input_filepaths: dict[Path, threading.Lock], outlier_proba_dir: Path, bounds: rasterio.coords.BoundingBox, res: float, crs: object, outlier_threshold: float, v_clip: float, progress_bar: tqdm.tqdm | None = None, vertical_shifts: pd.DataFrame | None = None):
    shape = adsvalbard.utilities.shape_from_bounds_res(bounds=bounds, res=[res] * 2)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, res, res)

    dtypes = {"median": "float32", "count": "uint8", "nmad": "float32", "doy": "uint16"}
    # output_paths = {"median": filepath}
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
                # BIGTIFF="YES",
                tiled=True,
                zlevel=12,
    )
    def write_params(kind: str) -> dict[str, object]:
        params = orig_params.copy()

        # params["fp"] = temp_paths[kind]
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
            doy = int(date_str[4:6]) * 30 + int(date_str[6:])
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

    # if np.count_nonzero(np.isfinite(median)) == 0:
    #     if progress_bar is not None:
    #         progress_bar.update()
    #     return

    # median[~np.isfinite(median)] = -9999

        # print(window.row_off, window.row_off + window.height, window.col_off,window.col_off + window.width)
        # if write_in_mem:
        #     for kind, arr in [("median", median), ("count", count), ("nmad", nmad)]:
        #         stacks[kind][
        #             window.row_off : window.row_off + window.height,
        #             window.col_off : window.col_off + window.width,
        #         ] = arr
        # else:
    for kind, arr in [("median", median), ("nmad", nmad), ("count", count), ("doy", doy)]:
        if dtypes[kind] == "float32":
            arr = np.where(np.isfinite(arr), arr, -9999)

        with rasterio.open(temp_paths[kind], "w", **write_params(kind=kind)) as raster:
            raster.write(arr, 1)

        shutil.move(temp_paths[kind], output_paths[kind])

            
        # if dtypes[kind] == "uint8":
        #     out_rasters[kind].write(arr, 1, window=window)
        # else:
        #     out_rasters[kind].write(, 1, window=window)

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

    if raster_type == "dhdt":
        raster_dir = temp_dir.joinpath("dhdt")
    elif raster_type == "dem":
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
    bad_titles = Path("temp.svalbard/bad_dems.txt").read_text().splitlines()

    vertical_shifts = None

    if raster_type == "dhdt":
        output_path = raster_dir / f"medians/{region}/dhdt/median_{filt_str}_dhdt{ext}.vrt"
        v_clip = 250 / (2021 - 2009)
        pattern = "*_dhdt.tif"
        # raster_files = []
        # for directory in dirs:
        #     raster_files += list(directory.glob(pattern))
    elif raster_type == "dem":
        output_path = temp_dir / f"medians/{region}/dem/median_{filt_str}_dem{ext}.vrt"
        v_clip = 1500
        pattern = "*_dem_coreg.tif"
        dirs = [raster_dir]
        bad_titles = list(set(bad_titles + Path("temp.svalbard/bad_dems_dem.txt").read_text().splitlines()))
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

        # print("SKIPPING REMOVAL OF EXTRA VERTCOREG BAD STRIPS")
        bad_titles = list(set(bad_titles + Path("temp.svalbard/bad_dems_dem_vertcoreg.txt").read_text().splitlines()))
        

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
    # centerpoint = 673467, 8866343 # Austfonna ETN-6
    # centerpoint = 709147, 8872572 # East Austfonna
    # centerpoint = 723077, 8872572 # East-east Austfonna
    def sort_chunk(chunk):

        easting = np.mean([chunk["bounds"].right, chunk["bounds"].left])
        northing = np.mean([chunk["bounds"].top, chunk["bounds"].bottom])

        return ((easting - centerpoint[0]) ** 2 + (northing - centerpoint[1]) ** 2) ** 2

    # def sort_chunk(chunk):

    #     easting = np.mean([chunk["bounds"].right, chunk["bounds"].left])
    #     northing = np.mean([chunk["bounds"].top, chunk["bounds"].bottom])

    #     return ((easting - 673248) ** 2 + (northing - 8901729) ** 2) ** 2
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
    # chunks_to_process.sort(key=sort_chunk)


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
                    # warnings.warn(f"{filepath} is missing.")
                    continue

                if filepath not in locks:
                    locks[filepath] = threading.Lock()

                # chunk["input_filepaths"][filepath] = locks[filepath]
                chunk["input_filepaths"][filepath] = None
                # chunk["input_filepaths"].append(c

                 
           # filepaths = Path(
           #
        
        # print(overlapping_strips.iloc[0])
        #
        # 
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

        # with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor, \
             # warnings.catch_warnings():
            # warnings.filterwarnings("ignore", message=".*All-NaN slice.*")
            # 
            # for _ in executor.map(process_chunk_wrapper, call_args):
            #     progress_bar.update(1)

    # with tqdm.tqdm(total=len(chunks_to_process)) as progress_bar:
    #     funcs = []
    #     for chunk in chunks_to_process:
    #         funcs.append(
    #             functools.partial(
    #                 process_chunk,
    #                 **chunk,
    #                 res=res,
    #                 crs=crs,
    #                 outlier_threshold=outlier_threshold,
    #                 v_clip=v_clip,
    #                 outlier_proba_dir=temp_dir / "outlier_proba",
    #                 progress_bar=progress_bar,
    #             )
    #         )

    #     if n_threads == 1 or len(chunks_to_process) == 1:
    #         for func in funcs:
    #             func()
    #     else:
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor,warnings.catch_warnings():
    #             warnings.filterwarnings("ignore", message=".*All-NaN slice.*")
    #             list(executor.map(lambda f: f(), funcs))


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
    bad_titles = Path("temp.svalbard/bad_dems.txt").read_text().splitlines()
    bad_coreg_titles = (temp_dir / "bad_dems_dem.txt").read_text().splitlines()
    bad_vertcoreg_titles = (temp_dir / "bad_dems_dem_vertcoreg.txt").read_text().splitlines()

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
            # if product != "dem_vertcoreg":
            #     continue
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


def hypsometric(bin_size: int = 5):
    import adsvalbard.outlines
    all_outlines = gpd.read_file("shapes/glacier_outlines.sqlite")
    all_outlines["area_km2"] = all_outlines["geometry"].area / 1e6
    all_outlines.sort_values("area_km2", inplace=True)

    dem_paths = {year: Path(f"temp.svalbard/stacks/median_dems_heerland/median_dem_heerland_{year}.tif") for year in range(2012, 2024)}

    replace_chars = {"ø": "o", "å": "a", " ": "_", "š": "s"}

    vals = []

    paths = {}
    for glacier, outlines in tqdm.tqdm(all_outlines.groupby("glac_name", sort=False)):
        if glacier != "Penckbreen":
            continue
        # print(outlines.iloc[0])
        glacier_id = str(glacier.lower())
        for c in replace_chars:
            glacier_id = glacier_id.replace(c, replace_chars[c])

        cache_path = Path(f"cache/hypsometric/hypso_{glacier_id}.nc")

        paths[glacier_id] = cache_path

        if cache_path.is_file():
            continue
        # if glacier != "Tinkarpbreen":
        #     continue
        masks = adsvalbard.outlines.generate_masks(outlines=outlines, frequency="YS")
        masks.coords["year"] = masks.coords["time"].dt.year
        masks = masks.swap_dims(time="year")

        masks.isel(year=-1).plot()
        plt.show()
        return

        with rasterio.open("temp/npi_mosaic.vrt") as raster:
            window = rasterio.windows.from_bounds(
                *masks.attrs["bounds"], raster.transform
            )
            npi_dem = raster.read(1, window=window, boundless=True, masked=True).filled(np.nan)

        elev_bins = np.linspace(np.nanmin(npi_dem), np.nanmax(npi_dem), bin_size + 1)
        elev_bins_orig = elev_bins.copy()

        elev_idx = np.digitize(npi_dem, elev_bins)

        del npi_dem

        out = xr.Dataset(coords={"elev_i": np.arange(elev_bins.shape[0] - 1), "year": list(dem_paths.keys())[1:]})
        out["elev"] = "elev_i", elev_bins_orig[1:] + (elev_bins_orig[1] - elev_bins_orig[0]) / 2

        for ext in ["dhdt", "dhdt2"]:
            for key in ["nmad", "median"]:
                out[f"{key}_{ext}"] = ("year", "elev_i"), np.empty((out["year"].shape[0], out["elev_i"].shape[0]), dtype="float32") + np.nan
            out[f"count_{ext}"] = ("year", "elev_i"), np.zeros((out["year"].shape[0], out["elev_i"].shape[0]), dtype="uint32")

        dem_prev = np.empty((0,))
        diff_prev = dem_prev.copy()
        for i, (year, filepath) in enumerate(dem_paths.items()):
            with rasterio.open(filepath) as raster:
                window = rasterio.windows.from_bounds(
                    *masks.attrs["bounds"], raster.transform
                )

                dem = raster.read(1, window=window, boundless=True, masked=True).filled(np.nan)

                if i == 0:
                    dem_prev = dem
                    continue

                diff = dem - dem_prev

                if i > 1:
                    acc = (diff - diff_prev) / 2
                else:
                    acc = diff_prev.copy()

                for idx in np.arange(1, elev_bins.shape[0]):

                    for arr, ext in [(diff, "dhdt"), (acc, "dhdt2")]:
                        if arr.shape[0] == 0:
                            continue

                        arr_sub = arr[elev_idx == idx]
                        arr_sub = arr_sub[np.isfinite(arr_sub)]

                        if arr_sub.shape[0] == 0:
                            continue

                        median = np.median(arr_sub)
                        key = {"year": year, "elev_i": idx - 1}
                        out[f"median_{ext}"].loc[key] = median
                        out[f"nmad_{ext}"].loc[key] = 1.4826 * np.median(np.abs(arr_sub - median))
                        out[f"count_{ext}"].loc[key] = arr_sub.shape[0]


                dem_prev = dem
                diff_prev = diff


        out["dh_offset"] = (out["median_dhdt"] - out["median_dhdt"].isel(elev_i=-1)).rolling(year=3, center=True, min_periods=1).mean()

        temp_path = cache_path.with_suffix(".nc.tmp")

        temp_path.parent.mkdir(exist_ok=True, parents=True)
        out.to_netcdf(temp_path, encoding={v: {"zlib": True, "complevel": 9} for v in out.data_vars}, engine="h5netcdf")
        shutil.move(temp_path, cache_path)
        # vals.append(out["dh_offset"].where(out["dh_offset"] >= 0).max().item())

        # print(out.set_coords("elev")["dh_offset"].swap_dims(elev_i="elev").plot(cmap="RdBu"))
        # plt.title(glacier)
        # plt.show()

    all_data = []
    for i, (glacier_id, filepath) in enumerate(paths.items()):
        with xr.open_dataset(filepath) as data:
            data = data.expand_dims("glacier_i")
            data.coords["glacier_i"] = "glacier_i", [i]

            data["glacier_id"] = "glacier_i", [glacier_id]

            all_data.append(data)

    data = xr.combine_nested(all_data, "glacier_i")

    print(data)

    max_dh = data["dh_offset"].max(["year", "elev_i"]).values
    max_dh = max_dh[max_dh > 0]

    plt.hist(np.log10(max_dh), bins=10)
    xticks = plt.gca().get_xticks()
    plt.xticks(xticks, np.round(10 ** xticks, 2))
    plt.show()

    print(max_dh)

    # plt.hist(vals, bins=10)
    # plt.show()



            

        

    
