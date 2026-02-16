import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from scipy.ndimage import distance_transform_edt

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
    t = np.arange(nt, dtype=data.dtype)
    t2 = t * t

    mask = np.isfinite(data)
    y = np.where(mask, data, 0.0)
    w = mask.astype(data.dtype)

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
        (S_w < 2) 
        # (var_t <= eps)# |
        # ~np.isfinite(r)
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
        print(nmad.dtype)
        for i in range(dems.shape[-1]):
            bad = (nmad > threshold) | (~np.isfinite(nmad))

            if np.count_nonzero(bad) == 0:
                break
            # print(f"Improving trend for {np.count_nonzero(bad)} pixels")

            bad_rows = np.where(bad)[0]

            dems_sub = dems[bad, :].copy()
            dems_sub[:, i] = np.nan
            new_slope, new_intercept, new_r, new_nmad = fit_trend_corr_nmad_1d(dems_sub)

            improved = np.isfinite(new_nmad) & (new_nmad < nmad[bad])

            if np.count_nonzero(improved) == 0:
                continue

            print(f"Improved trend for {np.count_nonzero(improved)} pixels by excluding {i} at {threshold=}")

            improved_rows = bad_rows[improved]

            slope[improved_rows] = new_slope[improved]
            intercept[improved_rows] = new_intercept[improved]
            r[improved_rows] = new_r[improved]
            nmad[improved_rows] = new_nmad[improved]

    return slope, intercept, r, nmad


def get_dem_chunk_path(kind: str, chunk_nr: str, chunksize: int, year: int) -> Path:
    stack_dir = Path("temp.svalbard/medians/svalbard/")

    # Translate e.g. p95 to 095
    if len(kind) == 3 and kind.startswith("p"):
        kind = "0" + kind[1:]

    if kind == "uncorr":
        return stack_dir / f"dem_noncoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif"

    return stack_dir / f"dem/median_filt_{kind}_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif"

def load_and_mosaic_dem(chunk_nr: str, chunksize: int, year: int, return_uncorr: bool = False):
    stack_dir = Path("temp.svalbard/medians/svalbard/")
    nmad_thresholds = {
        "p95": 10.,
        "p75": 30.,
        "uncorr": 30.,
    }
    paths = {kind: get_dem_chunk_path(kind, chunk_nr=chunk_nr, chunksize=chunksize, year=year) for kind in ["p95", "p75", "uncorr"]}
    # paths = {
    #     "p95": ,
    #     "uncorr": stack_dir / f"dem_noncoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif",
    # }
    # paths["p75"] = paths["p95"].parent.with_stem(paths["p95"].parent.stem.replace("095", "075")) / paths["p95"].name
    for key in list(paths.keys()):
        paths[f"{key}_nmad"] = paths[key].with_stem(paths[key].stem + "_nmad")

    with rio.open(paths["p95"]) as raster:
        # params["transform"] = raster.transform
        # params["crs"] = raster.crs
        meta = raster.meta
        quality_flag = np.ones((raster.height, raster.width), dtype="uint8")
        dem = np.empty(quality_flag.shape, dtype=raster.dtypes[0]) + np.nan
        nmad = dem.copy()
        count = quality_flag.copy()
        doy = quality_flag.copy().astype("uint16")
        xx, yy = np.meshgrid(np.linspace(raster.bounds.left, raster.bounds.right, raster.width), np.linspace(raster.bounds.bottom, raster.bounds.top, raster.height)[::-1])

    uncorr_dem = {}

    prev_mask = np.zeros(dem.shape, dtype=bool)
    for i, key in enumerate(["p75", "p95", "uncorr"], start=1):
        # if i == 3:
        #     continue
        with rio.open(paths[key].with_stem(paths[key].stem + "_nmad")) as raster:
            new_nmad = raster.read(1, masked=True).filled(np.nan)   

        mask = (~prev_mask) & (new_nmad < nmad_thresholds[key])

        if np.count_nonzero(mask) == 0:
            continue

        prev_mask = prev_mask | mask

        nmad[mask] = new_nmad[mask]
        quality_flag[mask] = i
        if return_uncorr:
            uncorr_dem["nmad"] = new_nmad

        # with rio.open(paths[key].with_stem(paths[key].stem + "_count")) as raster:
        #     arr = raster.read(1, masked=True).filled(0)
        #     count[mask] = arr[mask]
        #     if return_uncorr:
        #         uncorr_dem["count"] = arr

        # with rio.open(paths[key].with_stem(paths[key].stem + "_doy")) as raster:
        #     arr = raster.read(1, masked=True).filled(0)
        #     doy[mask] = arr[mask]
        #     if return_uncorr:
        #         uncorr_dem["doy"] = arr

        with rio.open(paths[key]) as raster:
            arr = raster.read(1, masked=True).filled(np.nan)
            dem[mask] = arr[mask]
            if return_uncorr:
                uncorr_dem["dem"] = arr

        if key == "uncorr":
            if return_uncorr:
                bias = get_uncorr_bias(year=year)(np.column_stack((xx.ravel(), yy.ravel())))
                uncorr_dem["dem"][...] += bias
                dem[mask] += bias[mask.ravel()]
            else:
                dem[mask] += get_uncorr_bias(year=year)(np.column_stack((xx.ravel(), yy.ravel()))[mask.ravel()])

    quality_flag[~np.isfinite(dem)] = 0

    return {
        "dem": dem,
        "quality_flag": quality_flag,
        "count": count,
        "nmad": nmad,
        "doy": doy,
        "meta": meta,
        "uncorr_dem": uncorr_dem,
    }

def write_formatted(key, fp, arr, params):
    shape = params["height"], params["width"]

    fp.parent.mkdir(exist_ok=True, parents=True)
    scale = 1
    offset = 0
    nodata = -9999.

    if arr.dtype in ["uint8", "uint16"]:
        nodata = 0

    if "_dem" in key or "nmad" in key or key.startswith("intercept_"):
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
        offset = -127
        arr = np.where(
            np.isfinite(arr),
            np.clip((arr * (umax / 1.)) - offset, a_min=0, a_max=umax),
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
            nodata,
        ).astype("int16")
        scale = vmax / umax
        

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


    # print(key, meta)
    print(f"Writing {fp}")
    with rio.open(fp, "w", **meta) as raster:
        raster.write(arr.reshape(shape), 1)
        raster.scales = (scale,)
        raster.offsets = (offset,)



def compute_weight_field(meta: dict[str, object], bad_regions: gpd.GeoDataFrame, R=1000.0):
    """
    Compute weight_B field for a tile, or return None if no blending is needed.

    Returns:
      - np.ndarray[float32] with shape (H, W) if blending is needed
      - None if no polygon is within distance R of the tile
    """
    gdf = bad_regions
    sindex = bad_regions.sindex
    transform = meta["transform"]
    height = meta["height"]
    width = meta["width"]
    # height, width = shape

    # Tile bounds in map coordinates
    minx, miny, maxx, maxy = array_bounds(height, width, transform)

    # Fast reject: search for polygons within R of tile bbox
    search_box = (minx - R, miny - R, maxx + R, maxy + R)
    candidate_idx = list(sindex.intersection(search_box))
    if not candidate_idx:
        # No polygons close enough → no blending
        return None

    polys = gdf.geometry.iloc[candidate_idx]

    # Pixel size (assume square pixels, north-up)
    pixel_size = abs(transform.a)

    # Margin in pixels for big window
    m = int(np.ceil(R / pixel_size))
    big_height = height + 2 * m
    big_width = width + 2 * m

    # Big transform: shift origin left and down by m pixels
    t = transform
    big_transform = Affine(
        t.a, t.b, t.c - m * t.a,
        t.d, t.e, t.f - m * t.e
    )

    # Rasterize polygons on big grid (1 inside, 0 outside)
    mask_inside_big = rasterize(
        [(geom, 1) for geom in polys],
        out_shape=(big_height, big_width),
        transform=big_transform,
        fill=0,
        dtype="uint8",
    )

    # Distance transform for outside pixels
    outside = (mask_inside_big == 0)
    distance_pixels = distance_transform_edt(outside)
    distance_m_big = distance_pixels * pixel_size

    # Crop back to tile extent
    mask_inside = mask_inside_big[m:m + height, m:m + width]
    distance_m = distance_m_big[m:m + height, m:m + width]

    # Compute weight_B
    weight_B = np.clip(1.0 - distance_m / R, 0.0, 1.0).astype("float32")
    weight_B[mask_inside == 1] = 1.0

    return weight_B


def load_polygons(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 32633:
        gdf = gdf.to_crs("EPSG:32633")
    sindex = gdf.sindex
    return gdf, sindex


def blend_bands(
    dem_a, count_a, nmad_a, doy_a,
    dem_b, count_b, nmad_b, doy_b,
    weight_B
):
    """
    Blend DEM, count, NMAD, DOY for one tile.

    Nodata assumptions:
      - dem_a, dem_b, nmad_a, nmad_b: NaN = nodata
      - count_a, count_b, doy_a, doy_b: 0 = nodata
      - Nodata masks are consistent across bands.
    """

    # DEM/NMAD valid mask from DEM (NaN-based)
    valid_a = np.isfinite(dem_a)
    valid_b = np.isfinite(dem_b)

    # --- DEM: NaN-aware weighted blend ---
    dem_out = np.full_like(dem_a, np.nan, dtype=dem_a.dtype)

    mask_both = valid_a & valid_b
    mask_a_only = valid_a & ~valid_b
    mask_b_only = valid_b & ~valid_a

    dem_out[mask_both] = (
        weight_B[mask_both] * dem_b[mask_both]
        + (1.0 - weight_B[mask_both]) * dem_a[mask_both]
    )
    dem_out[mask_a_only] = dem_a[mask_a_only]
    dem_out[mask_b_only] = dem_b[mask_b_only]

    # --- NMAD: same blending as DEM ---
    nmad_out = np.full_like(nmad_a, np.nan, dtype=nmad_a.dtype)

    nmad_out[mask_both] = (
        weight_B[mask_both] * nmad_b[mask_both]
        + (1.0 - weight_B[mask_both]) * nmad_a[mask_both]
    )
    nmad_out[mask_a_only] = nmad_a[mask_a_only]
    nmad_out[mask_b_only] = nmad_b[mask_b_only]

    # --- COUNT: uint8, use max where both valid ---
    count_out = np.zeros_like(count_a, dtype="uint8")

    # valid where DEM is valid (consistent masks)
    # if you prefer, you can additionally require count>0
    count_out[mask_a_only] = count_a[mask_a_only]
    count_out[mask_b_only] = count_b[mask_b_only]
    count_out[mask_both] = np.maximum(
        count_a[mask_both],
        count_b[mask_both]
    ).astype("uint8")

    # --- DOY: uint8, pick from dominant DEM ---
    doy_out = np.zeros_like(doy_a, dtype="uint16")

    # Only valid where DEM is valid
    # (assuming 0 = nodata)
    # Pixels valid only in A or only in B:
    doy_out[mask_a_only] = doy_a[mask_a_only]
    doy_out[mask_b_only] = doy_b[mask_b_only]

    # Where both valid: choose based on weight_B
    dominant_B = mask_both & (weight_B >= 0.5)
    dominant_A = mask_both & (weight_B < 0.5)

    doy_out[dominant_A] = doy_a[dominant_A]
    doy_out[dominant_B] = doy_b[dominant_B]

    return dem_out, count_out, nmad_out, doy_out


# -----------------------------------------------------------
# 4. Tile-level function tying everything together
# -----------------------------------------------------------

def blend_tile(tile_idx, gdf, sindex, R=1000.0):
    """
    Blend DEM A and DEM B products for one tile index.

    Returns:
      dem_out, count_out, nmad_out, doy_out, meta_out
    """

    # User-provided loading functions (already complex)
    dem_a, count_a, nmad_a, doy_a, meta_a = load_dem_a(tile_idx)
    dem_b, count_b, nmad_b, doy_b, meta_b = load_dem_b(tile_idx)

    # weight_B based on polygons and tile geo
    weight_B = compute_weight_field(
        meta=meta_a,
        shape=dem_a.shape,
        gdf=gdf,
        sindex=sindex,
        R=R
    )

    # Blend all bands
    dem_out, count_out, nmad_out, doy_out = blend_bands(
        dem_a, count_a, nmad_a, doy_a,
        dem_b, count_b, nmad_b, doy_b,
        weight_B
    )

    meta_out = meta_a  # same grid, dtype info can be adjusted if needed
    return dem_out, count_out, nmad_out, doy_out, meta_out



def mosaic_tile(chunk_nr: str = "018_011", chunksize: int = 512 * 7):

    out_dir = Path(f"temp.svalbard/filt/svalbard/chunks_{chunksize}/chunk_{chunk_nr}")
    years = list(range(2013, 2025))

    out_paths = {}
    for year in years:
        fp = out_dir / f"chunk_{chunk_nr}_{year}.tif"
        out_paths[f"{year}_dem"] = fp
        out_paths[f"{year}_quality"] = fp.with_stem(fp.stem + "_quality")
        out_paths[f"{year}_doy"] = fp.with_stem(fp.stem + "_doy")
        out_paths[f"{year}_count"] = fp.with_stem(fp.stem + "_count")
        out_paths[f"{year}_nmad"] = fp.with_stem(fp.stem + "_count")

    for trend, name in [("0", f"{years[0]}-{years[5]}"), ("1", f"{years[6]}-{years[-1]}"), ("full", f"{years[0]}-{years[-1]}")]:
        fp = out_dir / f"chunk_{chunk_nr}_trend_{name}.tif"
        out_paths[f"trend_{trend}"] = fp
        for part in ["r", "intercept", "nmad"]:
            out_paths[f"{part}_{trend}"] = fp.with_stem(fp.stem + f"_{part}")

    with rio.open(get_dem_chunk_path("p95", chunk_nr=chunk_nr, chunksize=chunksize, year=2024)) as raster:
        params = raster.meta

    bad_regions = gpd.read_file("shapes/bad_coreg_regions.geojson")
    # params = {}
    dems = []
    flags = []
    for year in years:
        print(year)

        weights = compute_weight_field(meta=params, bad_regions=bad_regions)

        out = load_and_mosaic_dem(chunk_nr=chunk_nr, chunksize=chunksize, year=year, return_uncorr=weights is not None)

        if weights is not None:
            out["dem"], out["count"], out["nmad"], out["doy"] = blend_bands(
                dem_a=out["dem"],
                count_a=out["count"],
                nmad_a=out["nmad"],
                doy_a=out["doy"],
                dem_b=out["uncorr"]["dem"],
                count_b=out["uncorr"]["count"],
                nmad_b=out["uncorr"]["nmad"],
                doy_b=out["uncorr"]["doy"],
                weight_B=weights,
            )
            out["quality_flag"][weights > 0] = 3

        dems.append(out["dem"].ravel()[:, None])

        flags.append(out["quality_flag"].ravel()[:, None])

        for key in ["nmad", "count", "doy"]:
            # write_formatted(
            #     key,
            #     out_paths[f"{year}_{key}"],
            #     out[key],
            #     params=params,
            # )
            del out[key]
        
    shape = params["height"], params["width"]
    dems = np.hstack(dems)
    flags = np.hstack(flags)

    # print("Removing bad timeseries")
    # dems[((flags <= 2).sum(axis=-1)[..., None] >= 2) & (flags == 3)] = np.nan

    data = {}

    print("Fitting trend")
    for (key, data_slice) in [("0", slice(0, 6)), ("1", slice(6, None)), ("full", None)]:
        data[f"trend_{key}"], data[f"intercept_{key}"], data[f"r_{key}"], data[f"nmad_{key}"] = fit_trend(
            dems[:, data_slice].copy() if data_slice is not None else dems,
            flags[:, data_slice].copy() if data_slice is not None else flags,
        )

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.subplots(1, 2, sharex=True, sharey=True)
        # axes[0].imshow(data[f"trend_{key}"].reshape(shape), vmin=-3, vmax=3, cmap="RdBu")
        # axes[1].imshow(data[f"nmad_{key}"].reshape(shape), vmin=0, vmax=10)
        # plt.show()
        
    # return
    # data["trend_0"], data["intercept_0"], data["r_0"], data["nmad_0"] = fit_trend(dems[:, :6].copy(), flags[:, :6].copy())
    # data["trend_1"], data["intercept_1"], data["r_1"], data["nmad_1"] = fit_trend(dems[:, 6:].copy(), flags[:, 6:].copy())

    # data["trend_full"], data["intercept_full"], data["r_full"], data["nmad_full"] = fit_trend(dems, flags)

    del dems
    del flags
    # for i, year in enumerate(years):
    #     data[f"{year}_dem"] = dems[:, i]
    #     data[f"{year}_quality"] = flags[:, i]

    
    for key, arr in data.items():
        fp = out_paths[key]
        write_formatted(key=key, fp=fp, arr=arr, params=params)
            
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


    

