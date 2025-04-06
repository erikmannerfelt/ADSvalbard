import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from pathlib import Path
import rasterio as rio
import rasterio.features
import warnings
import scipy.interpolate
import dask.array as da
import dask
import shapely
import pandas as pd
from tqdm.dask import TqdmCallback
import json
import tempfile
import shutil


def make_dhdt_ideal(per_year: xr.Dataset, bins: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*All-NaN slice.*")
        signal = (
            per_year["dhdt"]
            .where(per_year["glacier_mask_bool"])
            .median("year")
            .load()
            .stack(xy=["x", "y"])
            .groupby(per_year["digitized"].stack(xy=["x", "y"]))
            .median()
            .dropna("digitized")
        )

    # If there are too little data to make the signal, return the median dHdt instead.
    if signal.shape[0] == 0:
        return per_year["dhdt"].median("year")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
        signal = scipy.interpolate.interp1d(
            bins[signal.coords["digitized"].values] - np.diff(bins)[0],
            signal.values,
            fill_value="extrapolate",
        )

    ideal = signal(per_year["ref_dem"].where(per_year["glacier_mask_bool"]))

    return np.where(np.isfinite(ideal), ideal, 0)


def get_bounds(dataset: xr.Dataset) -> rio.coords.BoundingBox:
    res = float(dataset.x.isel(x=slice(2)).diff("x").isel(x=0))

    return rio.coords.BoundingBox(
        float(dataset.x.min() - res / 2),
        float(dataset.y.min() - res / 2),
        float(dataset.x.max() + res / 2),
        float(dataset.y.max() + res / 2),
    )


def generate_mask(stack: xr.Dataset, label: str, outlines: gpd.GeoDataFrame):
    output_path = Path(f"cache/{label}_mask.tif")

    if output_path.is_file():
        with rio.open(output_path) as raster:
            return raster.read(1)

    bounds = get_bounds(stack)

    shape = stack.y.shape[0], stack.x.shape[0]
    transform = rio.transform.from_bounds(*bounds, *shape[::-1])

    rasterized = rasterio.features.rasterize(
        ((row.geometry, i) for i, row in outlines.iterrows()),
        out_shape=shape,
        transform=transform,
    ).astype(int)

    with rio.open(
        output_path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=1,
        crs=rio.CRS.from_epsg(32633),
        transform=transform,
        dtype="uint32",
        compress="deflate",
        tiled=True,
    ) as raster:
        raster.write(rasterized, 1)

    output_path.parent.mkdir(exist_ok=True)

    return rasterized


NAMES = {
    "tinkarp": "Tinkarpbreen",
    "dronbreen": "Drønbreen",
}


def filter_glacier(
    subset: xr.Dataset,
    glacier_index: int,
    n_bins: int = 5,
    temporal_consistency_years: int = 1,
    temporal_consistency_threshold: float = 0.5,
    noise_floor_multiplier: float = 2.0,
) -> xr.DataArray:
    ref_dem_masked = subset["ref_dem"].where(subset["glacier_mask"]).load()
    subset["glacier_mask_bool"] = subset["glacier_mask"] == glacier_index

    bins = np.linspace(
        ref_dem_masked.min() * 0.9, ref_dem_masked.max() * 1.1, n_bins + 1
    )

    subset["digitized"] = ("y", "x"), np.digitize(ref_dem_masked, bins=bins)
    subset["dh"] = subset["ad_elevation"] - subset["ref_dem"]
    subset["dhdt"] = subset["dh"] / np.abs(
        subset["ref_dem_year"] - subset["time"].broadcast_like(subset["ad_elevation"])
    ).clip(min=1)

    years = np.unique(subset["year"])

    chunks = {k: v[0] for k, v in subset["dhdt"].chunksizes.items()}
    yearly = xr.DataArray(
        da.empty(
            dtype=subset["dhdt"].dtype,
            chunks=[chunks["year"], chunks["y"], chunks["x"]],
            shape=(years.size, subset["y"].shape[0], subset["x"].shape[0]),
        ),
        coords=[("year", years), subset["y"], subset["x"]],
        name="neighboring_dhdt",
    ).to_dataset()

    yearly["dhdt_ideal"] = yearly["neighboring_dhdt"].copy()

    make_dhdt_ideal_lazy = dask.delayed(make_dhdt_ideal)

    for year in years:
        yearly["neighboring_dhdt"].loc[{"year": year}] = (
            subset["dhdt"]
            .sel(
                year=slice(
                    year - temporal_consistency_years,
                    year + temporal_consistency_years + 1,
                )
            )
            .median("year")
        )

        ds = make_dhdt_ideal_lazy(
            subset.isel(year=np.argwhere(subset["year"].values == year).ravel()),
            bins=bins,
        )

        arr = da.from_delayed(
            ds,
            shape=(subset["y"].shape[0], subset["x"].shape[0]),
            dtype=subset["dhdt"].dtype,
        )

        yearly["dhdt_ideal"].loc[{"year": year}] = arr

    subset["temporally_consistent"] = (
        np.abs(subset["dhdt"] - yearly["neighboring_dhdt"].sel(year=subset["year"]))
        < temporal_consistency_threshold
    )

    subset["dhdt_ideal_res_abs"] = np.abs(
        subset["dhdt"] - yearly["dhdt_ideal"].sel(year=subset["year"])
    )

    subset["noise_level"] = noise_floor_multiplier * subset[
        "dhdt_ideal_res_abs"
    ].groupby("year").median(["x", "y"])
    subset["okay_noise"] = subset["dhdt_ideal_res_abs"] < subset["noise_level"]

    return ~(subset["okay_noise"] | subset["temporally_consistent"]).where(
        subset["glacier_mask_bool"], other=True
    )


def filter_stack(stack_path: Path, outline_path: Path | str, crs_epsg: int = 32633):
    output_path = Path("cache/").joinpath(stack_path.stem + "_filtered.nc")
    if output_path.is_file():
        return output_path

    label = stack_path.stem

    with xr.open_dataset(stack_path, chunks="auto") as stack:
        # Temporary subsetting to speed things up
        # stack = stack.sel(time=slice(2019, None))
        stack.coords["year"] = stack["time"].astype(int)
        stack = stack.swap_dims(time="year")

        bounds = get_bounds(stack)

        bbox_gpd = gpd.GeoSeries(shapely.geometry.box(*bounds), crs=crs_epsg)

        outlines = gpd.read_file(outline_path, bbox=bbox_gpd).to_crs(crs_epsg)

        outlines.geometry = outlines.geometry.transform(
            lambda geom: shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
        )
        outlines.index += 1

        stack.attrs["mask_ids"] = json.dumps(outlines["rgi_id"].to_dict())

        stack["glacier_mask"] = (
            ("y", "x"),
            generate_mask(stack=stack, label=label, outlines=outlines),
        )

        # Initialize the valid mask as if all values are valid
        stack["invalid_mask"] = (
            ("year", "y", "x"),
            da.zeros(
                dtype="bool",
                shape=stack["ad_elevation"].shape,
                chunks=stack["ad_elevation"].chunks,
            ),
        )

        for i, outline in outlines.iterrows():
            outline_bounds = rio.coords.BoundingBox(*outline.geometry.bounds)

            selection = dict(
                x=slice(outline_bounds.left, outline_bounds.right),
                y=slice(outline_bounds.top, outline_bounds.bottom),
            )

            result = filter_glacier(stack.sel(**selection), glacier_index=i)

            stack["invalid_mask"].loc[selection] = (
                stack["invalid_mask"].loc[selection] | result
            )

        stack = stack.swap_dims(year="time")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "data.nc"
            task = stack.to_netcdf(
                temp_path,
                encoding={k: {"zlib": True, "complevel": 5} for k in stack.data_vars},
                compute=False,
            )

            with TqdmCallback(desc="Filtering data"):
                task.compute()

            shutil.move(temp_path, output_path)

    return output_path


def main(label: str = "dronbreen"):
    outline_path = "zip://RGI2000-v7.0-G-07_svalbard_jan_mayen.zip/RGI2000-v7.0-G-07_svalbard_jan_mayen.shp"
    stack_path = Path(f"data/{label}.nc")

    filtered_path = filter_stack(stack_path, outline_path)

    with xr.open_dataset(filtered_path, chunks="auto") as stack:
        stack["ad_elevation"] = stack["ad_elevation"].where(~stack["invalid_mask"])
        stack["dh"] = stack["ad_elevation"] - stack["ref_dem"]
        stack["dhdt"] = stack["dh"] / np.abs(
            stack["ref_dem_year"] - stack["time"].broadcast_like(stack["ad_elevation"])
        ).clip(min=1)

        for year, data in stack.groupby(stack["time"].astype(int)):
            plt.title(year)
            data["dhdt"].median("time").plot(vmin=-2, vmax=2, cmap="RdBu")
            plt.show()


if __name__ == "__main__":
    main()
