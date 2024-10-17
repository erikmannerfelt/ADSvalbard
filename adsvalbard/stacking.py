from dask.array.core import PerformanceWarning
import xarray as xr
import dask
import pandas as pd
import numpy as np
import tqdm
import tqdm.dask
import zarr
import matplotlib.pyplot as plt
import geopandas as gpd
import scipy.interpolate

from pathlib import Path
import json
import warnings
import shutil

from adsvalbard.constants import CONSTANTS
import adsvalbard.utilities

def create_stack(region: str = "heerland") -> Path:

    # dem_paths = list(Path("temp.svalbard/heerland_dem_coreg/").glob("*/*dem_coreg.tif"))

    out_path = Path(f"temp.svalbard/stacks/dem_stack_{region}.zarr")
    temp_path = out_path.with_suffix(".zarr.tmp")

    if temp_path.is_dir():
        shutil.rmtree(temp_path)

    if out_path.is_dir():
        return out_path

    import rasterio.coords
    import rioxarray

    mask_stems = list(map(lambda fp: fp.stem.replace("_outlier_proba", ""), Path("temp.svalbard/outlier_proba/").glob("*/*.tif")))

    dem_paths = list(filter(lambda fp: fp.stem in mask_stems, Path(f"temp.svalbard/{region}_dem_coreg/").glob("*/*dem_coreg.tif")))


    bounds = CONSTANTS.regions[region]
    res = CONSTANTS.res
    shape = adsvalbard.utilities.get_shape(rasterio.coords.BoundingBox(**bounds), [res] * 2)

    xr_coords = {
        "y": np.linspace(bounds["bottom"] + res / 2, bounds["top"] - res / 2, shape[0])[::-1],
        "x": np.linspace(bounds["left"] + res / 2, bounds["right"] - res / 2, shape[1]),
    }


    dates = [(filepath, pd.to_datetime(filepath.stem.split("_")[3], format="%Y%m%d")) for filepath in dem_paths]
    dates.sort(key=lambda tup: tup[1])
    # dems = xr.Dataset(coords=xr_coords)

    # print(dems)
    # return

    dems = []
    for filepath, date in tqdm.tqdm(dates):
        mask_path = filepath.parent.parent.parent / f"outlier_proba/{filepath.parent.stem}/{filepath.stem}_outlier_proba.tif"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dem = rioxarray.open_rasterio(filepath, chunks="auto")
            mask = rioxarray.open_rasterio(mask_path, chunks="auto")


        mask = (
            mask
            .isel(band=0)
            .drop_vars(["band", "spatial_ref"])

        )
        
        dem = (
            dem
            .isel(band=0)
            .sel(x=slice(bounds["left"], bounds["right"]), y=slice(bounds["top"], bounds["bottom"]))
            .to_dataset(name="elevation")
            .drop_vars(["band", "spatial_ref"])
            .expand_dims(filename=[filepath.stem.replace("_dem_coreg", "")])
            .assign_coords(bounds=["xmin", "ymin", "xmax", "ymax"])
        )
        dem["elevation"].attrs = {}
        dem["bounding_box"] = ("filename", "bounds"), [[dem["x"].min(), dem["y"].min(), dem["x"].max(), dem["y"].max()]]

        with open(filepath.with_suffix(".json")) as infile:
            meta = json.load(infile)


        dem["stable_terrain_nmad"] = ("filename",), [meta["stable_nmad"]]
        dem["stable_terrain_frac"] = ("filename",), [meta["stable_fraction"]]
        dem["approx_vshift"] = ("filename",), [meta["steps"][0]["meta"]["_meta"]["matrix"][-2][-1]]
        dem["date"] = ("filename",), [date]

        dem["outlier_proba"] = mask

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            dems.append(dem.reindex(xr_coords, fill_value={"outlier_proba": np.uint8(255)}).chunk(x=512, y=512))

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
            data = rioxarray.open_rasterio(filepath, chunks="auto").isel(band=0).drop_vars(["band", "spatial_ref"])

        dems[key] = data.sel(**xr_coords).chunk(x=512, y=512)


    task = dems.to_zarr(temp_path, encoding={key: {"compressor": zarr.Blosc(cname="zstd", clevel=5, shuffle=2)} for key in ["elevation", "outlier_proba"]}, compute=False)

    with tqdm.dask.TqdmCallback(desc="Saving stack", smoothing=0.1):
        task.compute()


    shutil.move(temp_path, out_path)

    return out_path
    

def main():

    subsets = {
        "vallakra": {"xmin": 547000, "xmax": 554000, "ymin": 8640000, "ymax": 8648000},
        "tinkarp": {"xmin": 548762.5, "ymin": 8656585, "xmax": 553272.5, "ymax": 8658480},
    }

    stack_path = create_stack()

    glaciers = gpd.read_file("shapes/glacier_outlines.sqlite")

    glacier_name = "N Fredbreen"
    # glacier_name = "Tinkarpbreen"

    glacier = glaciers.query(f"glac_name == '{glacier_name}'")

    points = {
        "N Fredbreen": {
            "advancing_front": {"x": 545170, "y": 8626160},
            "scheele": {"x": 545783, "y": 8626293},
            "trend0": {"x": 543317.5, "y": 8627532.5}, 
            "trend1": {"x": 543312.5, "y": 8627532.5} 
        }
    }

    bounds = dict(zip(["xmin", "ymin", "xmax", "ymax"], glacier.geometry.buffer(500).total_bounds))

    with xr.open_zarr(stack_path) as data:

        # print(data["y"].max())

        # bounds = subsets["tinkarp"]

        data = data.sel(x=slice(bounds["xmin"], bounds["xmax"]), y=slice(bounds["ymax"], bounds["ymin"]))
        mask = (
            (data["bounding_box"].sel(bounds="xmin") < data["x"].max()) &
            (data["bounding_box"].sel(bounds="xmax") > data["x"].min()) &
            (data["bounding_box"].sel(bounds="ymin") < data["y"].max()) &
            (data["bounding_box"].sel(bounds="ymax") > data["y"].min())
        )
        overlapping = mask["filename"].where(mask.compute(), drop=True)
        data = data.sel(filename=overlapping)

        data["outlier_proba"] = data["outlier_proba"].astype("float32") / 255
        data["weight"] = 1 / data["outlier_proba"] ** 2

        # data["yearly_med_filt"] = data["elevation"].where(data["outlier_proba"] < 0.8).groupby(data["date"].dt.year).median()

        # data["elevation_count"] = data["elevation"].groupby(data["date"].dt.year).count()
        # enough_data = (data["elevation"].groupby(data["date"].dt.year).count() > 1)

        # data["yearly_med_filt"] = data["yearly_med_filt"].where(enough_data)



        data[["elevation", "outlier_proba"]] = data[["elevation", "outlier_proba"]].where((data["outlier_proba"] < 0.6))
        data["elevation_count"] = data["elevation"].groupby(data["date"].dt.year).count()
        enough_data = (data["elevation"].groupby(data["date"].dt.year).count() > 1)
        # data[["elevation", "outlier_proba"]] = data[["elevation", "outlier_proba"]].where((enough_data.sel(year=data["date"].dt.year)))
        data["yearly_med"] = data["elevation"].groupby(data["date"].dt.year).median()

        # data = data.assign_coords(date_yr=data["date"].dt.year + data["date"].dt.month / 12 + data["date"].dt.day / (12 * 30))
        data["date_yr"] = data["date"].dt.year + data["date"].dt.month / 12 + data["date"].dt.day / (12 * 30)
        data["yearly_med_date"] = data["date_yr"].load().groupby(data["date"].dt.year).median()


        # data["yearly_med"] = data["yearly_med"].chunk(year=-1).interpolate_na("year").compute()

        data["diff"] = data["yearly_med"].diff("year").compute()

        for i, (year, yearly) in enumerate(data["diff"].groupby("year", squeeze=False)):
            vals = yearly.values.squeeze()

            if np.count_nonzero(np.isfinite(vals)) == 0:
                continue

            if i == 0:
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

        poi_key = "scheele"
        point = data.sel(x=[points[glacier_name][poi_key]["x"]], y=[points[glacier_name][poi_key]["y"]], method="nearest")



        
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

        def interp_inner(arr: da.Array, times: np.ndarray, orig_shape: tuple[int, ...], years: np.ndarray):

            valid_mask = da.isfinite(arr)

            if da.count_nonzero(valid_mask) < 2:
                return da.zeros_like(years) + np.nan


            model = scipy.interpolate.interp1d(times[valid_mask], arr[valid_mask], fill_value="extrapolate")

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
            i = data["interp"].compute().load()

            print(i)

        i.sel(year=2022).plot()
        plt.show()
        
        # out = 

        return



        # print(point.swap_dims(filename="date").sortby("date").resample(date=pd.Timedelta(days=120)).map(lambda ds: ds.weighted(ds["weight"]).mean()))
        # return

        # point["elevation"].swap_dims(filename="date_yr").reset_coords(drop=True).plot.scatter()
        point.plot.scatter(x="date_yr", y="elevation", hue="outlier_proba")
        # print(point.swap_dims(filename="date").sortby("date").resample(date=pd.Timedelta(days=90)).mean())
        (
            point
            .swap_dims(filename="date")
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
        data["polyfit"] = data["elevation"].swap_dims(filename="date_yr").polyfit("date_yr", deg=1)["polyfit_coefficients"]

        data["poly_nmad"] = xr.polyval(data["date_yr"], data["polyfit"]).median("filename")

        data["poly_nmad"].plot()
        plt.show()
        print(data)

        # plt.hist(data["stable_terrain_nmad"], bins=50)
        # plt.show()
        
        # print(data)

