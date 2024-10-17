import xarray as xr
import numpy as np
from typing import Any
from pathlib import Path
import dask.array
import tempfile
import tqdm.dask
import shutil
import matplotlib.pyplot as plt
import math
import pandas as pd


def nmad(data: np.ndarray, nfact: float = 1.4826) -> np.floating[Any]:
    """Calculate the normalized median absolute deviation (NMAD) of an array.

    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. Höhle and Höhle (2009), http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    Parameters
    ----------
    data
        Input array
    nfact
        Normalization factor for the data

    Returns
    -------
    The (normalized) median absolute deviation of data.
    """
    data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))


def bin_data(filepath: Path) -> Path:

    out_path = filepath.with_stem("binned")

    if out_path.is_file():
        return out_path

    with xr.open_dataset(filepath, chunks={"time": 1, "y": -1, "x": -1}) as data:

        data["dh"] = data["ad_elevation"] - data["ref_dem"]
        data["stable_terrain"] = data["stable_terrain"] == 1
        data["rgi_mask"] = data["rgi_mask"] == 1

        data["ref_dem"].load()
        min_height, max_height = data["ref_dem"].where(data["rgi_mask"]).quantile([0., 1.], dim=["y", "x"]).load().values

        n_bins = 10
        approx_bin_size = np.round(((max_height - min_height) / n_bins) / 100) * 100.

        bins = np.arange(min_height, max_height + approx_bin_size, step=approx_bin_size)

        digitized = np.where(data["rgi_mask"], np.digitize(data["ref_dem"], bins=bins), -1)

        counts = np.unique(digitized, return_counts=True)[1][1:]
        data["digitized"] = ("y", "x"), digitized

        data["nmad"] = xr.apply_ufunc(nmad, data["dh"].where(data["stable_terrain"]), input_core_dims=[["y", "x"]], dask="parallelized", vectorize=True)
        data["dh"] -= xr.apply_ufunc(np.nanmedian, data["dh"].where(data["stable_terrain"]), input_core_dims=[["y", "x"]], dask="parallelized", vectorize=True)

        binned = xr.DataArray(
            dask.array.empty(
                shape=(data["time"].shape[0], bins.shape[0] - 1),
                chunks=(1, bins.shape[0] - 1),
                dtype="float32",
            ),
            coords=[data["time"], ("z", bins[1:] - np.diff(bins) / 2)]
        ).to_dataset(name="dh_median")

        binned["nmad"] = data["nmad"]
        binned["area"] = "z", counts * (5 * 5)

        for col in ["dh_count", "dh_lower", "dh_upper"]:
            binned[col] = ("time", "z"), dask.array.empty(
                shape=(data["time"].shape[0], bins.shape[0] - 1),
                chunks=(1, bins.shape[0] - 1),
                dtype="int32" if "count" in col else "float32"
            )


        for i in range(1, len(bins)):
            bin_val = binned["z"].isel(z=i - 1).item()

            dh_masked = data["dh"].where(data["digitized"] == i)

            binned["dh_median"].loc[{"z": bin_val}] = dh_masked.median(["y", "x"])
            binned["dh_count"].loc[{"z": bin_val}] = (~dh_masked.isnull()).sum(["y", "x"])

            quantiles = dh_masked.quantile([0.25, 0.75], ["y", "x"])

            for j, name in enumerate(["lower", "upper"]):
                binned[f"dh_{name}"].loc[{"z": bin_val}] = quantiles.isel(quantile=j)


        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "binned.nc"

            task = binned.to_netcdf(temp_path, encoding={v: {"zlib": True, "complevel": 5} for v in binned.data_vars}, compute=False)

            with tqdm.dask.TqdmCallback(desc="Binning stack", smoothing=0.1):
                task.compute()

            shutil.move(temp_path, out_path)

    return out_path



def main():
    filepath = Path("temp.haakonvii/glacier_stacks/kongsvegen/stack.nc")

    binned_path = bin_data(filepath)

    rng: np.random.Generator = np.random.default_rng(seed=1)

    with xr.open_dataset(binned_path) as data:

        data.coords["year"] = data["time"].astype(int)
        total_area = data["area"].sum("z").item()

        iters = []
        for year, year_data in data.groupby("year"):

            half_shape = max(int(year_data["year"].shape[0] / 2), 3)
            min_combinations = math.comb(year_data["year"].shape[0], half_shape)

            for i in range(min(min_combinations, 50)):
                subset_i = rng.integers(0, year_data["year"].shape[0], size=half_shape)
                year_data_sub = year_data.isel(time=subset_i)

                smb = 0.85 * (year_data_sub["dh_median"].median("time") * year_data_sub["area"]).sum("z") / total_area

                iters.append({"year": year, "i": i, "smb": smb.item()})

        smbs = pd.DataFrame.from_records(iters)

        grouped = smbs.groupby(["year"])["smb"]

        smb = grouped.median().to_frame("smb_median")
        smb["smb_upper"] = grouped.quantile(0.75) + 0.06
        smb["smb_lower"] = grouped.quantile(0.25) + 0.06

        smb.loc[2009, :] = 0
        smb.sort_index(inplace=True)

        smb.to_csv(filepath.with_name("smb.csv"))

        plt.fill_between(smb.index, smb["smb_lower"], smb["smb_upper"], alpha=0.3)
        plt.plot(smb["smb_median"])
        plt.scatter(smb.index, smb["smb_median"])
        plt.ylabel("Cumulative geodetic SMB since 2009 (m w.e.)")
        plt.show()

        return
        yearly = data.groupby("year").median()
        yearly["dh_count"] = data["dh_count"].groupby("year").sum()

        for suffix in ["_median", "_lower", "_upper"]:
            yearly[f"dv{suffix}"] = (yearly[f"dh{suffix}"] * yearly["area"]).sum("z")
            yearly[f"smb{suffix}"] = (yearly[f"dv{suffix}"] / yearly["area"].sum("z")) * 0.85

        plt.fill_between(yearly["year"], yearly["smb_median"] - yearly["nmad"], yearly["smb_median"] + yearly["nmad"], alpha=0.3)
        yearly["smb_median"].plot()
        yearly["smb_median"].plot.scatter()
        plt.show()




if __name__ == "__main__":
    main()
    
