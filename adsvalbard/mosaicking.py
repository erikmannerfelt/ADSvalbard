import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np

import scipy.interpolate

from pathlib import Path
import warnings


def sample_points_for_bias(year: int = 2024, n_points: int = 40000):
    cache_filepath = Path(f"temp.svalbard/sampled_points/sampled_points_for_bias_{year}.arrow")

    if cache_filepath.is_file():
        return gpd.read_feather(cache_filepath)
    land = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Land_f.shp").to_crs(32633)
    land = land[land["AREA"] > 1200000000].dissolve().simplify(100)

    bad_regions = gpd.read_file("shapes/bad_coreg_regions.geojson")
    land = land.difference(bad_regions.dissolve().geometry[0])

    dem_dir = Path(f"temp.svalbard/medians/svalbard/")
    rasters = {
        "corr": dem_dir / f"dem/median_filt_075_dem_{year}.vrt",
        "uncorr": dem_dir / f"dem_noncoreg/median_dem_{year}.vrt"
    }

    rasters["corr_nmad"] = rasters["corr"].with_stem(rasters["corr"].stem + "_nmad")
    rasters["uncorr_nmad"] = rasters["uncorr"].with_stem(rasters["uncorr"].stem + "_nmad")

    rng: np.random.Generator = np.random.default_rng(0)

    land_bounds = land.total_bounds

    x_pts = rng.uniform(land_bounds[0], land_bounds[2], size=n_points * 3)
    y_pts = rng.uniform(land_bounds[1], land_bounds[3], size=n_points * 3)

    points = gpd.points_from_xy(x_pts, y_pts, crs=land.crs)

    points = points[points.intersects(land.geometry[0])] 

    points = gpd.GeoDataFrame(geometry=points[:n_points])

    for key, fp in rasters.items():
        with rio.open(fp) as raster:
            arr = np.fromiter(map(lambda v: v[0], raster.sample(np.column_stack((points.geometry.x, points.geometry.y)))), count=points.shape[0], dtype=raster.dtypes[0])
            points[key] = np.where(arr == raster.nodata, np.nan, arr)

    points = points[(points["uncorr_nmad"] < 10) & (points["corr_nmad"] < 10)]

    points["diff"] = points["corr"] - points["uncorr"]
    # med = points["diff"].median()
    # points["diff"] -= med

    bin_size = 25000
    xbins = np.arange(land_bounds[0], land_bounds[2] + bin_size, bin_size)
    ybins = np.arange(land_bounds[1], land_bounds[3] + bin_size, bin_size)
    points["y_bin"] = np.digitize(points.geometry.y, ybins)
    points["x_bin"] = np.digitize(points.geometry.x, xbins)

    points["x"] = points.geometry.x
    points["y"] = points.geometry.y

    grid = []
    for _, subset in points.groupby("x_bin"):
        grid.append(subset.select_dtypes(np.number).groupby("y_bin").median())

    grid = pd.concat(grid)
    grid = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid["x"], grid["y"], crs=points.crs))

    cache_filepath.parent.mkdir(exist_ok=True ,parents=True)
    grid.to_feather(cache_filepath)
    return gpd.read_feather(cache_filepath)


def get_uncorr_bias(year: int = 2024):

    grid = sample_points_for_bias(year=year)
    model = scipy.interpolate.RBFInterpolator(grid[["x", "y"]], grid["diff"], kernel="linear")

    return model

    bounds = grid.buffer(50000).total_bounds
    xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[2], 500), np.linspace(bounds[1], bounds[3], 500)[::-1])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    axes[0].scatter(grid.geometry.x, grid.geometry.y, c=grid["diff"], vmin=-40, vmax=10)
    axes[1].imshow(model(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape), vmin=-40, vmax=10, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
    plt.show()

import numpy as np

def fit_trend_and_corr(data):
    """
    data: (nx, ny, nt), Δt = 1 along axis=-1
    Modifies data in-place: NaNs/inf -> 0.
    Returns slope, intercept, r (each (nx, ny)).
    """
    nt = data.shape[-1]
    t = np.arange(nt, dtype=data.dtype)
    t2 = t * t

    mask = np.isfinite(data)
    # data[~mask] = 0.0
    data = np.where(mask, data, 0.0)

    # w[~mask] = 0.
    w = mask.astype(float)

    # basic sums
    S_w  = w.sum(axis=-1)
    S_y  = (w * data).sum(axis=-1)
    S_yy = (w * data**2).sum(axis=-1)

    # time-weighted sums via tensordot (no big broadcasts)
    S_t  = np.tensordot(w,        t,  axes=([-1], [0]))
    S_tt = np.tensordot(w,        t2, axes=([-1], [0]))
    S_ty = np.tensordot(w * data, t,  axes=([-1], [0]))

    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = S_t / S_w
        y_mean = S_y / S_w

        var_t  = S_tt - S_t**2 / S_w
        var_y  = S_yy - S_y**2 / S_w
        cov_ty = S_ty - S_t * S_y / S_w

        slope = cov_ty / var_t
        intercept = y_mean - slope * t_mean
        r = cov_ty / np.sqrt(var_t * var_y)

    invalid = (
        (S_w < 2) |
        (var_t <= 0) |
        (var_y <= 0) |
        ~np.isfinite(r)
    )
    slope[invalid] = np.nan
    intercept[invalid] = np.nan
    r[invalid] = np.nan

    return slope, intercept, r

import numpy as np

def weighted_robust_trend(data, flags, outlier_sigma=3.0):
    """
    data:  (nx, ny, nt), may contain NaNs/infs
    flags: (nx, ny, nt), 1=good, 2=medium, 3=maybe bad (int or float)
    Δt = 1 along axis=-1.

    Returns:
        slope, intercept, r  (each (nx, ny))
    """

    nx, ny, nt = data.shape
    t = np.arange(nt, dtype=float)
    t2 = t * t

    # --- base weights from flags (you can tweak this mapping) ---
    # Example: 1 -> 1.0, 2 -> 0.5, 3 -> 0.1
    base_w = np.where(flags == 1, 1.0,
              np.where(flags == 2, 0.5,
              np.where(flags == 3, 0.1, 0.0)))

    # ignore non-finite data
    mask = np.isfinite(data)
    y = np.where(mask, data, 0.0)
    w = np.where(mask, base_w, 0.0)

    def fit_with_weights(y, w):
        # All shapes: y,w -> (nx,ny,nt)
        S_w  = w.sum(axis=-1)
        S_y  = (w * y).sum(axis=-1)
        S_yy = (w * y**2).sum(axis=-1)

        S_t  = np.tensordot(w,      t,  axes=([-1], [0]))
        S_tt = np.tensordot(w,      t2, axes=([-1], [0]))
        S_ty = np.tensordot(w * y,  t,  axes=([-1], [0]))

        with np.errstate(invalid="ignore", divide="ignore"):
            t_mean = S_t / S_w
            y_mean = S_y / S_w

            var_t  = S_tt - S_t**2 / S_w
            var_y  = S_yy - S_y**2 / S_w
            cov_ty = S_ty - S_t * S_y / S_w

            slope = cov_ty / var_t
            intercept = y_mean - slope * t_mean
            r = cov_ty / np.sqrt(var_t * var_y)

        invalid = (
            (S_w <= 0) |
            (var_t <= 0) |
            (var_y <= 0) |
            ~np.isfinite(r)
        )
        slope[invalid] = np.nan
        intercept[invalid] = np.nan
        r[invalid] = np.nan

        return slope, intercept, r

    # ---------- 1st pass: basic weighted least squares ----------
    slope1, intercept1, _ = fit_with_weights(y, w)

    # ---------- Outlier detection (per-pixel, over time) ----------
    # residuals: (nx,ny,nt)
    res = y - (slope1[..., None] * t + intercept1[..., None])

    # Only consider points with positive base weight
    valid = (w > 0)
    # simple robust scale: per-pixel std of residuals over time
    # (could swap for MAD if you want more robustness)
    with np.errstate(invalid="ignore"):
        # compute std ignoring zero-weighted points
        res2 = np.where(valid, res, np.nan)
        sigma = np.nanstd(res2, axis=-1)    # (nx,ny)

    # avoid zero sigma
    sigma = np.where(sigma <= 0, np.nan, sigma)

    # mark outliers: |res| > outlier_sigma * sigma
    # sigma[..., None] broadcasts over time
    outliers = np.abs(res) > (outlier_sigma * sigma[..., None])

    # down-weight or drop outliers
    # simplest: drop them completely
    w2 = np.where(outliers, 0.0, w)

    # ---------- 2nd pass: refit with updated weights ----------
    slope, intercept, r = fit_with_weights(y, w2)

    return slope, intercept, r
import numpy as np

def fit_trend_and_corr_1d(data):
    """
    data: array of shape (nx, nt)
          NaNs / infs are ignored.
    Δt = 1 along axis=-1.

    Returns:
        slope   : (nx,)
        intercept : (nx,)
        r       : (nx,)  Pearson correlation between t and data
    """
    nx, nt = data.shape
    t = np.arange(nt, dtype=float)
    t2 = t * t

    # mask of finite values
    mask = np.isfinite(data)
    y = np.where(mask, data, 0.0)
    w = mask.astype(float)        # 0/1 weights

    # 1D sums along time
    S_w  = w.sum(axis=-1)                     # (nx,)
    S_y  = (w * y).sum(axis=-1)              # (nx,)
    S_yy = (w * y**2).sum(axis=-1)           # (nx,)

    # time‑weighted sums
    S_t  = np.dot(w,  t)                     # (nx,)
    S_tt = np.dot(w,  t2)                    # (nx,)
    S_ty = np.dot(w * y, t)                  # (nx,)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = S_t / S_w
        y_mean = S_y / S_w

        var_t  = S_tt - S_t**2 / S_w
        var_y  = S_yy - S_y**2 / S_w
        cov_ty = S_ty - S_t * S_y / S_w

        slope = cov_ty / var_t
        intercept = y_mean - slope * t_mean
        r = cov_ty / np.sqrt(var_t * var_y)

    invalid = (
        (S_w < 2) |
        (var_t <= 0) |
        (var_y <= 0) |
        ~np.isfinite(r)
    )
    slope[invalid] = np.nan
    intercept[invalid] = np.nan
    r[invalid] = np.nan

    return slope, intercept, r

# import numpy as np

# def fit_trend_corr_nmad_1d(data, eps=1e-12):
#     """
#     data: array of shape (nx, nt)
#           NaNs / infs are ignored.
#     Δt = 1 along axis=-1.

#     Returns:
#         slope     : (nx,)
#         intercept : (nx,)
#         r         : (nx,)  Pearson correlation between t and data
#         nmad      : (nx,)  NMAD of residuals
#     """
#     nx, nt = data.shape
#     t = np.arange(nt, dtype=data.dtype)
#     t2 = t * t

#     # mask of finite values
#     mask = np.isfinite(data)
#     y = np.where(mask, data, 0.0)
#     w = mask.astype(data.dtype)        # 0/1 weights

#     # 1D sums along time
#     S_w  = w.sum(axis=-1)                     # (nx,)
#     S_y  = (w * y).sum(axis=-1)              # (nx,)
#     S_yy = (w * y**2).sum(axis=-1)           # (nx,)

#     # time‑weighted sums
#     S_t  = np.dot(w,  t)                     # (nx,)
#     S_tt = np.dot(w,  t2)                    # (nx,)
#     S_ty = np.dot(w * y, t)                  # (nx,)

#     with np.errstate(invalid="ignore", divide="ignore"):
#         t_mean = S_t / S_w
#         y_mean = S_y / S_w

#         var_t  = S_tt - S_t**2 / S_w
#         var_y  = S_yy - S_y**2 / S_w
#         cov_ty = S_ty - S_t * S_y / S_w

#         slope = cov_ty / var_t
#         intercept = y_mean - slope * t_mean
#         r = cov_ty / np.sqrt(var_t * var_y)

#         # residuals for NMAD
#         # model prediction: shape (nx, nt)
#         y_fit = slope[:, None] * t + intercept[:, None]
#         res = np.where(mask, y - y_fit, np.nan)   # ignore invalid points

#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         # per‑row robust sigma via NMAD
#         med_res = np.nanmedian(res, axis=-1)      # (nx,)
#         abs_dev = np.abs(res - med_res[:, None])
#         nmad = 1.4826 * np.nanmedian(abs_dev, axis=-1)

#     invalid = (
#         (S_w < 2) |
#         (var_t <= eps) |
#         (var_y <= eps)
#         # (var_y <= 0)# |
#         # ~np.isfinite(r)
#     )
#     slope[invalid] = np.nan
#     intercept[invalid] = np.nan
#     r[invalid] = np.nan
#     nmad[invalid] = np.nan

#     return slope, intercept, r, nmad

# import numpy as np

def fit_trend_corr_nmad_1d(data, eps=1e-12):
    """
    data: array of shape (nx, nt)
          NaNs / infs are ignored.
    Δt = 1 along axis=-1.

    Returns:
        slope     : (nx,)
        intercept : (nx,)
        r         : (nx,)  Pearson correlation between t and data
        nmad      : (nx,)  NMAD of residuals
    """
    nx, nt = data.shape
    t = np.arange(nt, dtype=float)
    t2 = t * t

    mask = np.isfinite(data)
    y = np.where(mask, data, 0.0)
    w = mask.astype(float)

    S_w  = w.sum(axis=-1)           # (nx,)
    S_y  = (w * y).sum(axis=-1)
    S_yy = (w * y**2).sum(axis=-1)

    S_t  = np.dot(w,  t)
    S_tt = np.dot(w,  t2)
    S_ty = np.dot(w * y, t)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = S_t / S_w
        y_mean = S_y / S_w

        var_t  = S_tt - S_t**2 / S_w
        var_y  = S_yy - S_y**2 / S_w
        cov_ty = S_ty - S_t * S_y / S_w

        slope = cov_ty / var_t
        intercept = y_mean - slope * t_mean
        r = cov_ty / np.sqrt(var_t * var_y)

    # --- Handle flat series (var_y ~ 0) explicitly ---
    flat = (S_w >= 2) & (var_t > eps) & (np.abs(var_y) <= eps)
    slope[flat] = 0.0
    intercept[flat] = y_mean[flat]
    r[flat] = 0.0

    # residuals for NMAD
    y_fit = slope[:, None] * t + intercept[:, None]
    res = np.where(mask, y - y_fit, np.nan)

    has_res = np.isfinite(res).any(axis=-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
    # with np.errstate(all="ignore"):
        med_res = np.nanmedian(res, axis=-1)
        abs_dev = np.abs(res - med_res[:, None])
        nmad = 1.4826 * np.nanmedian(abs_dev, axis=-1)

    nmad[~has_res] = np.nan

    # --- Only mark as invalid when really no usable trend info ---
    invalid = (
        (S_w < 2) |
        (var_t <= eps) |
        ~np.isfinite(r)
    )
    slope[invalid] = np.nan
    intercept[invalid] = np.nan
    r[invalid] = np.nan
    nmad[invalid] = np.nan

    return slope, intercept, r, nmad


def fit_trend(dems, flags):
    dems[((flags <= 2).sum(axis=-1)[..., None] >= 2) & (flags == 3)] = np.nan
    slope, intercept, r, nmad = fit_trend_corr_nmad_1d(dems)

    for threshold in [1.5, 4]:
        for i in range(dems.shape[-1]):
            bad = (nmad > threshold) | (~np.isfinite(nmad))

            if np.count_nonzero(bad) == 0:
                break
            print(f"Improving trend for {np.count_nonzero(bad)} pixels")

            bad_rows = np.where(bad)[0]

            dems_sub = dems[bad, :].copy()
            dems_sub[:, i] = np.nan
            new_slope, new_intercept, new_r, new_nmad = fit_trend_corr_nmad_1d(dems_sub)

            improved = np.isfinite(new_nmad) & (new_nmad < nmad[bad])

            if np.count_nonzero(improved) == 0:
                continue

            improved_rows = bad_rows[improved]

            slope[improved_rows] = new_slope[improved]
            intercept[improved_rows] = new_intercept[improved]
            r[improved_rows] = new_r[improved]
            nmad[improved_rows] = new_nmad[improved]

    return slope, intercept, r, nmad

def load_and_mosaic_dem(chunk_nr: str, chunksize: int, year: int):
    stack_dir = Path("temp.svalbard/medians/svalbard/")
    nmad_thresholds = {
        "p95": 10.,
        "p75": 30.,
        "uncorr": 30.,
    }
    paths = {
        "p95": stack_dir / f"dem/median_filt_095_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif",
        "uncorr": stack_dir / f"dem_noncoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif",
    }
    paths["p75"] = paths["p95"].parent.with_stem(paths["p95"].parent.stem.replace("095", "075")) / paths["p95"].name
    for key in list(paths.keys()):
        paths[f"{key}_nmad"] = paths[key].with_stem(paths[key].stem + "_nmad")

    with rio.open(paths["p95"]) as raster:
        # params["transform"] = raster.transform
        # params["crs"] = raster.crs
        meta = raster.meta
        quality_flag = np.ones((raster.height, raster.width), dtype="uint8")
        dem = np.empty(quality_flag.shape, dtype=raster.dtypes[0]) + np.nan
        xx, yy = np.meshgrid(np.linspace(raster.bounds.left, raster.bounds.right, raster.width), np.linspace(raster.bounds.bottom, raster.bounds.top, raster.height)[::-1])

    for i, key in enumerate(["p75", "p95", "uncorr"], start=1):
        # if i == 3:
        #     continue
        with rio.open(paths[f"{key}_nmad"]) as raster:
            nmad = raster.read(1, masked=True).filled(1000)   

        with rio.open(paths[key]) as raster:
            mask = (~np.isfinite(dem)) & (nmad < nmad_thresholds[key])
            dem = np.where(
                mask,
                raster.read(1, masked=True).filled(np.nan),
                dem,
            )
            quality_flag[mask] = i

        if key == "uncorr":
            dem[mask] += get_uncorr_bias(year=year)(np.column_stack((xx.ravel(), yy.ravel()))[mask.ravel()])

    quality_flag[~np.isfinite(dem)] = 0

    return {
        "dem": dem,
        "quality_flag": quality_flag,
        "meta": meta,
    }

def mosaic_tile(chunk_nr: str = "018_011", chunksize: int = 512 * 7):

    out_dir = Path(f"temp.svalbard/filt/svalbard/chunks_{chunksize}/chunk_{chunk_nr}")
    years = list(range(2013, 2025))

    out_paths = {}
    for year in years:
        fp = out_dir / f"chunk_{chunk_nr}_{year}.tif"
        out_paths[f"{year}_dem"] = fp
        out_paths[f"{year}_quality"] = fp.with_stem(fp.stem + "_quality")

    for trend, name in [("0", f"{years[0]}-{years[5]}"), ("1", f"{years[6]}-{years[-1]}"), ("full", f"{years[0]}-{years[-1]}")]:
        fp = out_dir / f"chunk_{chunk_nr}_trend_{name}.tif"
        out_paths[f"trend_{trend}"] = fp
        for part in ["r", "intercept", "nmad"]:
            out_paths[f"{part}_{trend}"] = fp.with_stem(fp.stem + f"_{part}")

        
    # if all(fp.is_file() for fp in out_paths.values()):
    #     return

    # for key in out_paths:
    #     print(key, out_paths[key])
    # return

    nmad_thresholds = {
        "p95": 10.,
        "p75": 30.,
        "uncorr": 30.,
    }
    stack_dir = Path("temp.svalbard/medians/svalbard/")

    params = {}

    years = []
    dems = []
    flags = []

    for year in range(2013, 2025):
        print(year)
        # if year != 2017:
        #     continue
        paths = {
            "p95": stack_dir / f"dem/median_filt_095_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif",
            "uncorr": stack_dir / f"dem_noncoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif",
        }
        paths["p75"] = paths["p95"].parent.with_stem(paths["p95"].parent.stem.replace("095", "075")) / paths["p95"].name
        for key in list(paths.keys()):
            paths[f"{key}_nmad"] = paths[key].with_stem(paths[key].stem + "_nmad")

        with rio.open(paths["p95"]) as raster:
            params["transform"] = raster.transform
            params["crs"] = raster.crs
            quality_flag = np.ones((raster.height, raster.width), dtype="uint8")
            dem = np.empty(quality_flag.shape, dtype=raster.dtypes[0]) + np.nan
            xx, yy = np.meshgrid(np.linspace(raster.bounds.left, raster.bounds.right, raster.width), np.linspace(raster.bounds.bottom, raster.bounds.top, raster.height)[::-1])

        for i, key in enumerate(["p75", "p95", "uncorr"], start=1):
            # if i == 3:
            #     continue
            with rio.open(paths[f"{key}_nmad"]) as raster:
                nmad = raster.read(1, masked=True).filled(1000)   

            with rio.open(paths[key]) as raster:
                mask = (~np.isfinite(dem)) & (nmad < nmad_thresholds[key])
                dem = np.where(
                    mask,
                    raster.read(1, masked=True).filled(np.nan),
                    dem,
                )
                quality_flag[mask] = i

            if key == "uncorr":
                dem[mask] += get_uncorr_bias(year=year)(np.column_stack((xx.ravel(), yy.ravel()))[mask.ravel()])

        quality_flag[~np.isfinite(dem)] = 0

        years.append(year)
        flags.append(quality_flag)
        dems.append(dem)

        # import matplotlib.pyplot as plt
        # plt.imshow(dem)
        # plt.show()


    shape = dems[0].shape
    dems = np.reshape(dems, (len(dems), -1)).T
    flags = np.reshape(flags, (len(flags), -1)).T

    # print("Removing bad timeseries")
    # dems[((flags <= 2).sum(axis=-1)[..., None] >= 2) & (flags == 3)] = np.nan

    data = {}

    print("Fitting trend")
    data["trend_0"], data["intercept_0"], data["r_0"], data["nmad_0"] = fit_trend(dems[:, :6].copy(), flags[:, :6].copy())
    data["trend_1"], data["intercept_1"], data["r_1"], data["nmad_1"] = fit_trend(dems[:, 6:].copy(), flags[:, 6:].copy())

    data["trend_full"], data["intercept_full"], data["r_full"], data["nmad_full"] = fit_trend(dems.copy(), flags.copy())

    for i, year in enumerate(years):
        data[f"{year}_dem"] = dems[:, i]
        data[f"{year}_quality"] = flags[:, i]

    
    for key, arr in data.items():
        fp = out_paths[key]

        fp.parent.mkdir(exist_ok=True, parents=True)
        scale = 1
        offset = 0
        nodata = -9999.

        if arr.dtype == "uint8":
            nodata = 255

        if "_dem" in key or key.startswith("nmad_") or key.startswith("intercept_"):
            umax = 65534
            vmax = 100. if key.startswith("nmad_") else 1500.
            arr = np.where(
                np.isfinite(arr),
                np.clip((arr * (umax / vmax)), a_min=0, a_max=umax),
                umax + 1
            ).astype("uint16")
            scale = vmax / umax
            nodata = umax + 1
        elif key.startswith("r_"):
            umax = 127
            offset = 127
            arr = np.where(
                np.isfinite(arr),
                np.clip((arr * (umax / 1.)) + 127, a_min=0, a_max=umax),
                umax + 1
            ).astype("uint8")
            scale = 1. / umax
            nodata = umax + 1
        elif key.startswith("trend_"):
            info = np.iinfo(np.int16)
            nodata = info.min
            umin = info.min + 1
            umax = info.max - 1
            vmax = 300

            arr = np.where(
                np.isfinite(arr),
                np.clip((arr * (umax / vmax)), a_min=umin, a_max=umax),
                umax + 1
            ).astype("int16")
            scale = vmax / umax
            
            
        elif "_quality" in key:
            nodata = 0



        if arr.dtype == "float64":
            arr = arr.astype("float32")

        meta = params | {
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "tiled": True,
            "compress": "deflate",
            "zlevel": 9,
            "dtype": str(arr.dtype),
            "nodata": nodata,
        }

        if meta["dtype"] == "float32":
            meta["predictor"] = 3
        else:
            meta["predictor"] = 2

        # if meta["dtype"] == "uint8":
        #     meta["nodata"] = 0
        # elif meta["dtype"] == "uint16":
        #     meta["nodata"] = 65535


        print(key, meta)
        with rio.open(fp, "w", **meta) as raster:
            raster.write(arr.reshape(shape), 1)
            raster.scales = (scale,)
            raster.offsets = (offset,)
            
    return


    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(slope.reshape(shape), vmin=-3, vmax=3, cmap="RdBu")
    axes[1].imshow(nmad.reshape(shape), vmin=0, vmax=10)
    plt.show()
    
    print(dems.shape)
    print(dems.shape)
    return
    dems = np.moveaxis(dems, 0, 2)
    flags = np.moveaxis(flags, 0, 2)

    print("Removing bad timeseries")
    dems[(flags <= 2).sum(axis=-1)[:, :, None] & (flags == 3)] = np.nan

    print("Fitting trend")
    slope, intercept, r= fit_trend_and_corr(dems)



        

        

def mosaic_timeseries():

    ...


    

