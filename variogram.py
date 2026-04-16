import xdem
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import pandas as pd


def main():
    # print(xdem.version.version)
    # return

    # bounds = rio.coords.BoundingBox(left=504602, bottom=8645983, right=544085, top=8673150)
    bounds = rio.coords.BoundingBox(left=499302, bottom=8649321, right=552749, top=8686096)

    dhdt = xdem.DEM("temp.svalbard/medians/svalbard/dem/dhdt_2024.tif", load_data=False)
    dhdt.crop(bounds)

    dhdt *= (2024 - 2009)
    stable = xdem.DEM("temp/stable_terrain.tif", load_data=False)
    stable.crop(dhdt)

    dhdt.data.mask[stable.data == 0] = True

    dhdt_pts = dhdt.to_points(subset=40000).ds.dropna(subset=["b1"])
    dhdt_pts["x"] = dhdt_pts.geometry.x - dhdt_pts.geometry.x.min()
    dhdt_pts["y"] = dhdt_pts.geometry.y - dhdt_pts.geometry.y.min()

    # bin_func = np.linspace(50, 10000, 10)
    # bin_func = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 2000, 3000, 5000, 7000, 9000, 10000]
    bin_func = 10 ** np.linspace(np.log10(30), np.log10(5000), 50) 
    # np.random.seed(3)
    variogram = xdem.spatialstats.sample_empirical_variogram(
        values=dhdt_pts["b1"].values,
        coords=dhdt_pts[["x", "y"]].values,
        subsample=5000,
        subsample_method="cdist_point",
        n_variograms=30,
        # runs=None,
        verbose=True,
        bin_func=bin_func,
        n_jobs=1,
        random_seed=3,
    )
    print(variogram)

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], variogram)

    areas = 10 ** np.linspace(0, np.log10(30000e6), 1000)
    neff = []
    for area in areas:
        neff.append(xdem.spatialstats.neff_circular_approx_numerical(area, params))

    pd.Series(neff, pd.Index(areas, name="neff")).to_csv("vgm_neff_cirq.csv")

    variogram["bin_width"] = np.r_[[0], np.diff(variogram.index)]
    fig = plt.figure(figsize=(8.3 * 0.81, 5 * 0.81))
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[0.3, 0.7])
    axes[0].bar(variogram.index, variogram["count"], width=variogram["bin_width"] * 1.1, align="edge", color="gray")
    axes[1].plot(variogram.index, vgm_model(variogram.index), label="Variogram model", color="black")
    axes[1].errorbar(variogram.index, variogram["exp"], yerr=variogram["err_exp"], fmt="x", label="Empirical variogram", color="royalblue")

    axes[1].set_ylim(0, 5)
    for r in params["range"].values:
        axes[1].vlines(r, *axes[1].get_ylim(), color="gray", linestyles="--")
        
    
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Bin count")
    axes[1].set_ylabel("Variance (m²)")
    axes[1].legend(loc="lower right")
    axes[1].set_xlabel("Spatial lag (m)")


    print(params)

    plt.tight_layout()
    # xdem.spatialstats.plot_variogram(
    #     variogram,
    #     xscale_range_split=[100, 1000],
    #     list_fit_fun=[vgm_model],
    #     list_fit_fun_label=["Standardized double-range variogram"],
    # )
    # xdem.spatialstats.
    # dhdt.show()
    plt.savefig("figures/ad_variogram.pdf")
    plt.show()


if __name__ == "__main__":
    main()
