import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from scipy.ndimage import distance_transform_edt
import tqdm.contrib.concurrent
import shutil

import scipy.interpolate

from pathlib import Path
import warnings

def sample_raster(filepath: Path, geometry: gpd.GeoSeries, nodata: float | int = np.nan) -> np.ndarray:
    with rio.open(filepath) as raster:
        arr = np.fromiter(map(lambda v: v[0], raster.sample(np.column_stack((geometry.x, geometry.y)))), count=geometry.shape[0], dtype=raster.dtypes[0])
        return np.where(arr == raster.nodata, nodata, arr)
    

def sample_points_for_bias(year: int, n_points: int = 80000, point_density_threshold: float | None = None, verbose: bool = True, redo: bool= False):

    if point_density_threshold is None:
        # This year had particularly annoying outliers that require this threshold instead
        if year == 2022:
            point_density_threshold = 0.27
        elif year == 2023:
            point_density_threshold = 0.2
        else:
            point_density_threshold = 0.1
    cache_filepath = Path(f"temp.svalbard/sampled_points/sampled_points_for_bias_{year}.arrow")

    if cache_filepath.is_file() and not redo:
        return gpd.read_feather(cache_filepath)
    if verbose:
        print("Reading land outlines")
    land = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Land_f.shp").simplify(20).to_frame().dissolve().to_crs(32633)
    # land = land[land["AREA"] > 1200000000].dissolve().simplify(100)
    # land = land.dissolve().simplify(20)

    # The stable terrain mask erroneously includes some glaciers as of February 2026.
    if verbose:
        print("Reading glacier outlines")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        glaciers = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Isbreer_f.shp").simplify(20).to_frame().dissolve().to_crs(32633)

    # import matplotlib.pyplot as plt
    # glaciers.plot()
    # plt.show()
    # return

    bad_regions = gpd.read_file("shapes/bad_coreg_regions.geojson")
    # land = land.difference(bad_regions.dissolve().geometry[0])

    dem_dir = Path(f"temp.svalbard/medians/svalbard/")
    rasters = {
        "corr": dem_dir / f"dem/median_filt_075_dem_{year}.vrt",
        "uncorr": dem_dir / f"dem_noncoreg/median_dem_{year}.vrt"
    }

    rasters["corr_nmad"] = rasters["corr"].with_stem(rasters["corr"].stem + "_nmad")
    rasters["uncorr_nmad"] = rasters["uncorr"].with_stem(rasters["uncorr"].stem + "_nmad")

    rng: np.random.Generator = np.random.default_rng(0)

    land_bounds = glaciers.total_bounds

    x_pts = rng.uniform(land_bounds[0], land_bounds[2], size=n_points * 3)
    y_pts = rng.uniform(land_bounds[1], land_bounds[3], size=n_points * 3)

    points = gpd.points_from_xy(x_pts, y_pts, crs=land.crs)

    points = points[points.intersects(land.geometry[0])] 

    points = gpd.GeoDataFrame(geometry=points[:n_points])

    if verbose:
        print("Sampling stable terrain")
    points["stable"] = sample_raster("temp/stable_terrain.tif", geometry=points.geometry, nodata=0) == 1

    bad_stable = points.loc[points["stable"], "geometry"].intersects(glaciers.dissolve().geometry[0])
    points.loc[bad_stable[bad_stable].index, "stable"] = False
    

    # import matplotlib.pyplot as plt
    # # points = points[~points["npi_dem"].isna()]
    # plt.scatter(points.geometry.x, points.geometry.y, c=points["stable"].astype(int))
    # plt.show()

    if verbose:
        print("Sampling NPI mosaic")
    points.loc[points["stable"], "npi_dem"] = sample_raster("temp/npi_mosaic.vrt", geometry=points.loc[points["stable"], "geometry"])
    
    bad_unstable = points.loc[~points["stable"], "geometry"].intersects(bad_regions.dissolve().geometry[0])
    # Hacky solution to avoid sampling the corrected ArcticDEM here.
    points.loc[bad_unstable[bad_unstable].index, "stable"] = True

    # print(points)
    # import matplotlib.pyplot as plt
    # points = points[~points["npi_dem"].isna()]
    # plt.scatter(points.geometry.x, points.geometry.y, c=points["npi_dem"].astype(int))
    # plt.show()

    for key, fp in rasters.items():
        if verbose:
            print(f"Sampling {fp.name}")
        if "uncorr" in key:
            points[key] = sample_raster(fp, geometry=points.geometry)
        else:
            points.loc[~points["stable"], key] = sample_raster(fp, geometry=points.loc[~points["stable"], "geometry"])

    points = points[points["uncorr_nmad"] < 10]

    good_npi = ~points["npi_dem"].isna()
    good_corr = (points["corr_nmad"] < 10) & ~good_npi
    
    points.loc[good_npi, "diff"] = points.loc[good_npi, "npi_dem"] - points.loc[good_npi, "uncorr"]
    points.loc[good_corr, "diff"] = points.loc[good_corr, "npi_dem"] - points.loc[good_npi, "uncorr"]

    points = points[~points["diff"].isna()]
    # points = points[(points["corr_nmad"]
    # points = points[((points["uncorr_nmad"] < 10) & (points["corr_nmad"] < 10)) | (points["uncorr_nmad"] < 10)]

    # points["diff"] = points["corr"] - points["uncorr"]

    bin_size = 30000
    xbins = np.arange(land_bounds[0], land_bounds[2] + bin_size, bin_size)
    ybins = np.arange(land_bounds[1], land_bounds[3] + bin_size, bin_size)
    points["y_bin"] = np.digitize(points.geometry.y, ybins)
    points["x_bin"] = np.digitize(points.geometry.x, xbins)

    points["x"] = points.geometry.x
    points["y"] = points.geometry.y

    grid = []
    for _, subset in points.groupby("x_bin"):
        grouped = subset.select_dtypes(np.number).groupby("y_bin")
        new = grouped.median()
        new["count"] = grouped["diff"].count()
        grid.append(new)

    grid = pd.concat(grid)
    grid = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid["x"], grid["y"], crs=points.crs))
    grid["count"] /= grid["count"].median()

    # Remove points whose point counts are less than X% of the median count. These are often really off.
    # grid = grid[(grid["count"] / grid["count"].median()) > point_density_threshold]
    grid = grid[grid["count"] > point_density_threshold]

    cache_filepath.parent.mkdir(exist_ok=True ,parents=True)
    grid.to_feather(cache_filepath)
    return gpd.read_feather(cache_filepath)


def get_uncorr_bias(year: int):

    grid = sample_points_for_bias(year=year)
    model = scipy.interpolate.RBFInterpolator(grid[["x", "y"]], grid["diff"], kernel="linear")

    return model

    bounds = grid.buffer(50000).total_bounds
    xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[2], 500), np.linspace(bounds[1], bounds[3], 500)[::-1])
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    axes[0].scatter(grid.geometry.x, grid.geometry.y, c=grid["diff"], vmin=-40, vmax=-20)
    axes[1].imshow(model(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape), vmin=-40, vmax=-20, extent=[bounds[0], bounds[2], bounds[1], bounds[3]])

    # grid["count"] /= grid["count"].median()
    axes[2].scatter(grid.geometry.x, grid.geometry.y, c=grid["count"], vmin=0, vmax=0.5)
    for _, point in grid.iterrows():
        axes[2].annotate(f"{point['count']:.2f}",(point.geometry.x, point.geometry.y))
        
    plt.show()
    return model

def fit_trend_corr_nmad_1d(data, eps=1e-12) -> dict[str, np.ndarray]:
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
    t = np.arange(data.shape[1], dtype=data.dtype)

    mask = np.isfinite(data)
    y = np.where(mask, data, 0.0)
    w = mask.astype(data.dtype)

    S_w  = w.sum(axis=-1)           # (nx,)
    S_y  = (w * y).sum(axis=-1)
    S_yy = (w * y**2).sum(axis=-1)

    S_t  = np.dot(w,  t)
    S_tt = np.dot(w,  t ** 2)
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

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "nmad": nmad,
        "var_t": var_t,
    }


def fit_trend(dems, flags, verbose: bool = True) -> dict[str, np.ndarray]:
    # If there are at least three good DEM points, remove all the quality=3 points
    dems[((np.isin(flags, [1, 2, 4])).sum(axis=-1)[..., None] >= 3) & (flags == 3)] = np.nan
    fit = fit_trend_corr_nmad_1d(dems)

    for threshold in [1.5]:#, 4]:
        if verbose:
            print(f"Improving with threshold ", threshold)
        for i in range(dems.shape[-1]):
            bad = (fit["nmad"] > threshold) | (~np.isfinite(fit["nmad"]))

            if np.count_nonzero(bad) == 0:
                break

            bad_rows = np.where(bad)[0]

            dems_sub = dems[bad, :].copy()
            dems_sub[:, i] = np.nan
            new = fit_trend_corr_nmad_1d(dems_sub)

            improved = np.isfinite(new["nmad"]) & (new["nmad"] < fit["nmad"][bad])

            if np.count_nonzero(improved) == 0:
                continue

            if verbose:
                n_improved =np.count_nonzero(improved)
                print(f"Improved trend for {n_improved} pixels by excluding year{i}.")

            improved_rows = bad_rows[improved]

            for col in fit:
                fit[col][improved_rows] = new[col][improved]

    return fit


def get_dem_chunk_path(kind: str, chunk_nr: str, chunksize: int, year: int) -> Path:
    stack_dir = Path("temp.svalbard/medians/svalbard/")

    # Translate e.g. p95 to 095
    if len(kind) == 3 and kind.startswith("p"):
        kind = "0" + kind[1:]

    if kind == "uncorr":
        return stack_dir / f"dem_noncoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif"
    if kind == "vertcoreg":
        return stack_dir / f"dem_vertcoreg/median_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif"
        

    return stack_dir / f"dem/median_filt_{kind}_dem_{year}_chunks_{chunksize}/chunk_{chunk_nr}.tif"

def load_and_mosaic_dem(chunk_nr: str, chunksize: int, year: int, return_uncorr: bool = False, min_uncorr_count: int = 3, min_vertcorr_count: int = 1, use_vertcoreg: bool = True):
    nmad_thresholds = {
        "p75": 10.,
        "p95": 30.,
        "uncorr": 30.,
        "vertcoreg": 10.,
    }
    paths = {kind: get_dem_chunk_path(kind, chunk_nr=chunk_nr, chunksize=chunksize, year=year) for kind in ["p95", "p75", "uncorr", "vertcoreg"]}
    for key in list(paths.keys()):
        paths[f"{key}_nmad"] = paths[key].with_stem(paths[key].stem + "_nmad")

    with rio.open(paths["p75"]) as raster:
        meta = raster.meta
        quality_flag = np.ones((raster.height, raster.width), dtype="uint8")
        dem = np.full(quality_flag.shape, np.nan, dtype=raster.dtypes[0])
        nmad = dem.copy()
        count = quality_flag.copy() - 1
        doy = quality_flag.copy().astype("uint16")
        xy_coords = np.dstack(np.meshgrid(np.linspace(raster.bounds.left, raster.bounds.right, raster.width), np.linspace(raster.bounds.bottom, raster.bounds.top, raster.height)[::-1])).reshape((-1, 2))

    ocean = read_coastline(year=str(year), meta=meta)

    uncorr_dem = {}

    prev_mask = np.zeros(dem.shape, dtype=bool)
    vertcoreg_exists = False
    for i, key in [(4, "vertcoreg"), (1, "p75"), (2, "p95"), (3, "uncorr")]:
        if key == "vertcoreg" and not use_vertcoreg:
            continue
        elif key == "vertcoreg":
            if not paths[key].is_file():
                continue
            vertcoreg_exists = True
        else:
            # If there is a vertcoreg tile, only use vertcoreg and uncorr
            if key != "uncorr" and vertcoreg_exists:
                continue
        if key == "p95" and not paths[key].is_file():
            continue
        with rio.open(paths[key].with_stem(paths[key].stem + "_nmad")) as raster:
            new_nmad = raster.read(1, masked=True).filled(np.nan)   

        mask = (~prev_mask) & (new_nmad < nmad_thresholds[key])

        with rio.open(paths[key].with_stem(paths[key].stem + "_count")) as raster:
            arr = raster.read(1, masked=True).filled(0)
            if key in ["uncorr", "vertcoreg"]:
                good_count = arr >= (min_uncorr_count if key == "uncorr" else min_vertcorr_count)
                mask = mask & good_count
                arr[~good_count] = 0
            count[mask] = arr[mask]
            if return_uncorr:
                uncorr_dem["count"] = arr

        with rio.open(paths[key].with_stem(paths[key].stem + "_doy")) as raster:
            arr = raster.read(1, masked=True).filled(0)
            doy[mask] = arr[mask]
            if return_uncorr:
                uncorr_dem["doy"] = arr

        with rio.open(paths[key]) as raster:
            arr = raster.read(1, masked=True).filled(np.nan)
            dem[mask] = arr[mask]
            if return_uncorr:
                uncorr_dem["dem"] = arr

        if key == "uncorr":
            if return_uncorr:
                bias = get_uncorr_bias(year=year)(xy_coords)
                uncorr_dem["dem"].ravel()[:] += bias
                dem[mask] += bias[mask.ravel()]
            else:
                dem[mask] += get_uncorr_bias(year=year)(xy_coords[mask.ravel()])

        nmad[mask] = new_nmad[mask]
        quality_flag[mask] = i
        if return_uncorr:
            uncorr_dem["nmad"] = new_nmad
        prev_mask = prev_mask | mask

        # If there's a vertcoreg tile, use ONLY that
        # if key == "vertcoreg":
        #     break

        # if np.count_nonzero(~prev_mask) == 0:
        #     break

    quality_flag[~np.isfinite(dem)] = 0

    return {
        "dem": dem,
        "quality": quality_flag,
        "count": count,
        "nmad": nmad,
        "doy": doy,
        "ocean": ocean,
        "meta": meta,
        "uncorr": uncorr_dem,
    }

def write_formatted(key, fp, arr, params, verbose: bool = True, redo: bool = False):
    shape = params["height"], params["width"]

    if fp.is_file() and not redo:
        return

    fp.parent.mkdir(exist_ok=True, parents=True)
    scale = 1
    offset = 0
    nodata = -9999.

    if arr.dtype in ["uint8", "uint16"]:
        nodata = 0

    if arr.dtype == "bool":
        nodata = 255
        arr = arr.astype("uint8")

    if key == "dem" or "nmad" in key or key.startswith("intercept_") or key.startswith("se_"):
        umax = 65534
        if key.startswith("nmad_"):
            vmax = 100.
        elif key.startswith("se_"):
            vmax = 10.
        else:
            vmax = 1500.
        arr = np.where(
            np.isfinite(arr),
            np.clip((arr * (umax / vmax)), a_min=0, a_max=umax),
            umax + 1
        ).astype("uint16")
        scale = vmax / umax
        nodata = umax + 1
    elif key.startswith("r_"):
        umax = 254
        offset = -1
        arr = np.where(
            np.isfinite(arr),
            np.clip(((arr + 1) * (umax / 2.)), a_min=0, a_max=umax),
            umax + 1
        ).astype("uint8")
        scale = 2. / umax
        nodata = umax + 1
    elif key.startswith("slope_") or key == "accel":
        info = np.iinfo(np.int16)
        nodata = info.min
        umin = info.min + 1
        umax = info.max - 1
        vmax = 20 if key == "accel" else 50

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
        "driver": "GTiff",
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
    if verbose:
        print(f"Writing {fp}")
    tmp_fp = fp.with_suffix(".tmp")
    with rio.open(tmp_fp, "w", **meta) as raster:
        raster.write(arr.reshape(shape), 1)
        raster.scales = (scale,)
        raster.offsets = (offset,)
    shutil.move(tmp_fp, fp)



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


def get_mosaic_filenames(chunk_nr: str, chunksize: int = 512 * 7) -> dict[str, Path]:
    years = list(range(2013, 2025))
    out_dir = Path(f"temp.svalbard/filt/svalbard/chunks_{chunksize}/chunk_{chunk_nr}")
    out_paths = {}
    for year in years:
        fp = out_dir / f"chunk_{chunk_nr}_{year}.tif"
        out_paths[f"{year}_dem"] = fp
        for key in ["quality", "doy", "count", "nmad", "ocean"]:
            out_paths[f"{year}_{key}"] = fp.with_stem(fp.stem + f"_{key}")

    for trend, name in [("0", f"{years[0]}-{years[5]}"), ("1", f"{years[6]}-{years[-1]}"), ("full", f"{years[0]}-{years[-1]}")]:
        fp = out_dir / f"chunk_{chunk_nr}_trend_{name}.tif"
        for part in ["r", "intercept", "nmad", "slope", "se"]:
            out_paths[f"{part}_{trend}"] = fp.with_stem(fp.stem + f"_{part}")

        if trend == "full":
            out_paths["accel"] = fp.with_stem(fp.stem + "_accel")
            out_paths["se_accel"] = fp.with_stem(fp.stem + "_accel_se")

    out_paths["quality_mode"] = out_dir / f"chunk_{chunk_nr}_quality_mode.tif"

    return out_paths

def read_coastline(year: str, meta: dict[str, object]) -> np.ndarray:
    import rasterio.features
    import shapely.geometry

    # import time

    # start_time = time.time()
    filepath = Path(f"data/coastlines/Coast{year}.zip")

    bounds = shapely.geometry.box(*rio.transform.array_bounds(meta["height"], meta["width"], meta["transform"]))
    coast = gpd.read_file(filepath, bbox=bounds).to_crs(meta["crs"])

    try:
        arr = rasterio.features.rasterize(coast.geometry, out_shape=(meta["height"], meta["width"]), transform=meta["transform"])
    except ValueError as e:
        if not "No valid geometry" in str(e):
            raise
        return np.zeros((meta["height"], meta["width"])) == 0

    # print(f"Took {time.time() - start_time:.1f}s")

    return arr == 0

def mosaic_tile(chunk_nr: str = "007_004", chunksize: int = 512 * 7, verbose: bool = True, redo: bool = False):

    years = list(range(2013, 2025))
    out_paths = get_mosaic_filenames(chunk_nr=chunk_nr, chunksize=chunksize)

    if all(fp.is_file() for fp in out_paths.values()) and not redo:
        return out_paths
    # out_dir = Path(f"temp.svalbard/filt/svalbard/chunks_{chunksize}/chunk_{chunk_nr}")

    # out_paths = {}
    # for year in years:
    #     fp = out_dir / f"chunk_{chunk_nr}_{year}.tif"
    #     out_paths[f"{year}_dem"] = fp
    #     for key in ["quality", "doy", "count", "nmad"]:
    #         out_paths[f"{year}_{key}"] = fp.with_stem(fp.stem + f"_{key}")

    # for trend, name in [("0", f"{years[0]}-{years[5]}"), ("1", f"{years[6]}-{years[-1]}"), ("full", f"{years[0]}-{years[-1]}")]:
    #     fp = out_dir / f"chunk_{chunk_nr}_trend_{name}.tif"
    #     for part in ["r", "intercept", "nmad", "slope", "se"]:
    #         out_paths[f"{part}_{trend}"] = fp.with_stem(fp.stem + f"_{part}")

    #     if trend == "full":
    #         out_paths["accel"] = fp.with_stem(fp.stem + "_accel")
    #         out_paths["se_accel"] = fp.with_stem(fp.stem + "_accel_se")

    # Open a raster of the chunk (could be any) to get its metadata
    with rio.open(get_dem_chunk_path("p75", chunk_nr=chunk_nr, chunksize=chunksize, year=2024)) as raster:
        params = raster.meta

    bad_regions = gpd.read_file("shapes/bad_coreg_regions.geojson")
    dems = []
    flags = []
    oceans = []
    weights = compute_weight_field(meta=params, bad_regions=bad_regions, R=2000)

    # Exception for Storøya (use the normal coreg results)
    if chunk_nr == "003_022":
        weights = None

    # import matplotlib.pyplot as plt
    # plt.imshow(weights)
    # plt.show()
    # return

    for year in years:
        if verbose:
            print(year)

        # Load and mosaic the different DEM qualities. If vertcoreg tiles exist, use them.
        out = load_and_mosaic_dem(chunk_nr=chunk_nr, chunksize=chunksize, year=year, use_vertcoreg=weights is not None)
        # If the chunk is close to a bad region, it will have weights to blend between vertcoreg and normal coreg:
        if weights is not None:
            # import matplotlib.pyplot as plt
            # plt.imshow(weights.reshape((params["height"], params["width"])))
            # plt.show()
            # If the chunk is completely within a bad coreg region, then the vertcoreg will already have been loaded
            # and taken precedence. Therefore, nothing more needs to be done
            # If the chunk is at the boundary, all weights will not be 1 and blending is required
            if not np.all(weights == 1.):
                # Load the data again but without vertcoreg
                nonvertcoreg = load_and_mosaic_dem(chunk_nr=chunk_nr, chunksize=chunksize, year=year, use_vertcoreg=False)
                # Blend between vertcoreg and non-vertcoreg
                out["dem"], out["count"], out["nmad"], out["doy"] = blend_bands(
                    dem_a=nonvertcoreg["dem"],
                    count_a=nonvertcoreg["count"],
                    nmad_a=nonvertcoreg["nmad"],
                    doy_a=nonvertcoreg["doy"],
                    dem_b=out["dem"],
                    count_b=out["count"],
                    nmad_b=out["nmad"],
                    doy_b=out["doy"],
                    weight_B=weights,
                )
                out["quality"][weights == 0.] = nonvertcoreg["quality"][weights == 0.]
                del nonvertcoreg
                
        # if np.any(out["quality"] == 4) and weights is not None:
        #     print("PRELININARY DISABLING WEIGHTS")
        #     weights = None

        # if weights is not None:
        #     out["dem"], out["count"], out["nmad"], out["doy"] = blend_bands(
        #         dem_a=out["dem"],
        #         count_a=out["count"],
        #         nmad_a=out["nmad"],
        #         doy_a=out["doy"],
        #         dem_b=out["uncorr"]["dem"],
        #         count_b=out["uncorr"]["count"],
        #         nmad_b=out["uncorr"]["nmad"],
        #         doy_b=out["uncorr"]["doy"],
        #         weight_B=weights,
        #     )
        #     out["quality"][weights > 0] = 3

        dems.append(out["dem"].ravel()[:, None])
        flags.append(out["quality"].ravel()[:, None])
        oceans.append(out["ocean"].ravel()[:, None])

        for key in ["nmad", "count", "doy", "dem", "quality", "ocean"]:
            write_formatted(
                key,
                out_paths[f"{year}_{key}"],
                out[key],
                params=params,
                verbose=verbose,
                redo=redo,
            )
            del out[key]
            # if key not in ["dem", "quality"]:
            #     del out[key]
    dems = np.hstack(dems)
    flags = np.hstack(flags)
    oceans = np.hstack(oceans)

    quality_counts = np.stack([(flags == v).sum(axis=1) for v in range(1, 5)], axis=1)
    quality_mode = (quality_counts.argmax(axis=1) + 1).astype("uint8")   # values 1..4
    quality_mode[quality_counts.sum(axis=1) == 0] = 0  # rows that were all zeros

    # These points are ocean in every DEM. Should be skipped
    only_ocean = np.count_nonzero(~oceans, axis=1) < 2
    # only_ocean = oceans.all(axis=-1)

    # Retain points that aren't only in the ocean
    dems = dems[~only_ocean]

    # Apply the yearly ocean mask
    dems[oceans[~only_ocean]] = 0.
    flags = flags[~only_ocean]
    quality_mode = quality_mode[~only_ocean]

    # enough_pts = np.count_nonzero(np.isfinite(dems), axis=-1) >= 2
    # dems = dems[enough_pts]
    # flags = flags[enough_pts]

    del oceans
    
    data = {"quality_mode": quality_mode}

    if verbose:
        print("Fitting trend")
    for (key, data_slice) in [("0", slice(0, 6)), ("1", slice(6, None)), ("full", None)]:

        fit = fit_trend(
            dems[:, data_slice].copy() if data_slice is not None else dems,
            flags[:, data_slice].copy() if data_slice is not None else flags,
            verbose=verbose,
        )

        for key2 in fit:
            data[f"{key2}_{key}"] = fit[key2]
            # data[f"_{key}"], data[f"intercept_{key}"], data[f"r_{key}"], data[f"nmad_{key}"] = 

        data[f"se_{key}"] = np.full_like(data[f"slope_{key}"], np.nan, dtype=data[f"slope_{key}"].dtype)

        valid = np.isfinite(data[f"nmad_{key}"]) & np.isfinite(data[f"var_t_{key}"]) & (data[f"var_t_{key}"] > 1e-12)

        data[f"se_{key}"][valid] = data[f"nmad_{key}"][valid] / np.sqrt(data[f"var_t_{key}"][valid])

        del data[f"var_t_{key}"]

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # axes = fig.subplots(1, 2, sharex=True, sharey=True)
        # axes[0].imshow(data[f"trend_{key}"].reshape(shape), vmin=-3, vmax=3, cmap="RdBu")
        # axes[1].imshow(data[f"nmad_{key}"].reshape(shape), vmin=0, vmax=10)
        # plt.show()
        

    del dems
    del flags

    # It's 6 years in between interval midpoints, so it should be divided by 6
    data["accel"] = (data["slope_1"] - data["slope_0"]) / 6
    data["se_accel"] = np.sqrt(data["se_0"] ** 2 + data["se_1"] ** 2) / 6

    # mask_valid = np.zeros_like(only_ocean, dtype=bool)
    # mask_valid[~only_ocean] = enough_pts          # where we have enough points

    # mask_no_enough = np.zeros_like(only_ocean, dtype=bool)
    # mask_no_enough[~only_ocean] = ~enough_pts     # where we don't
    
    for key, arr in data.items():
        if key not in out_paths:
            continue

        arr2 = np.zeros((only_ocean.shape[0],), dtype=arr.dtype)

        arr2[~only_ocean] = arr

        # arr2[~only_ocean] = np.where(
        #     enough_pts,
        #     arr,
        #     np.nan if "float" in str(arr.dtype) else 0,
        # )

        # arr2[~only_ocean][enough_pts] = arr
        # arr2[~only_ocean][~enough_pts] = np.nan if "float" in str(arr.dtype) else 0
        # arr2[~only_ocean] = arr
        fp = out_paths[key]
        write_formatted(key=key, fp=fp, arr=arr2, params=params, verbose=verbose, redo=redo)
            
    return out_paths


    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # axes = fig.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(slope.reshape(shape), vmin=-3, vmax=3, cmap="RdBu")
    # axes[1].imshow(nmad.reshape(shape), vmin=0, vmax=10)
    # plt.show()
    
    # print(dems.shape)
    # print(dems.shape)
    # return
    # dems = np.moveaxis(dems, 0, 2)
    # flags = np.moveaxis(flags, 0, 2)

    # print("Removing bad timeseries")
    # dems[(flags <= 2).sum(axis=-1)[:, :, None] & (flags == 3)] = np.nan

    # print("Fitting trend")
    # slope, intercept, r= fit_trend_and_corr(dems)



def process_chunk_wrapper(args, verbose: bool = False):
    return mosaic_tile(**args, verbose=verbose)

def mosaic_all_tiles(chunksize: int = 512 * 7, n_workers: int | None = 8, redo = False):

    # chunk_nrs = [fp.stem.replace("chunk_", "").replace("_count", "") for fp in Path(f"temp.svalbard/medians/svalbard/dem/median_filt_075_dem_2024_chunks_{chunksize}/").glob("chunk_*count.tif")]

    import adsvalbard.stacking

    chunk_outlines = adsvalbard.stacking.make_chunk_polygons()
    chunk_outlines["chunk_nr"] = chunk_outlines["chunk_id"].str.replace("chunk_", "")

    # Hacky way to get only the chunks that exist in all years
    chunk_nrs = []
    for year in range(2013, 2025):
        for raster_type in ["dem", "dem_noncoreg"]:
            # print(f"{raster_type}: {year}")
            filt = [0.75, 0.95] if raster_type == "dem" else [0.75]

            for threshold in filt:
                new_paths = adsvalbard.stacking.create_median_stack(years=year, n_threads=2,raster_type=raster_type, outlier_threshold=threshold)
                new_chunk_nrs = [fp.stem.replace("chunk_", "") for fp in new_paths]

                if len(chunk_nrs) == 0:
                    chunk_nrs = new_chunk_nrs
                else:
                    chunk_nrs = [ch for ch in new_chunk_nrs if ch in chunk_nrs]
    import os

    # return
    chunk_nrs.sort()

    call_args = []
    all_out_paths = []
    for chunk_nr in chunk_nrs:

        if chunk_nr not in chunk_outlines["chunk_nr"].values:
            continue

        parts = chunk_nr.split("_")

        # Do only Austfonna
        # if (parts[0] <= "001") or (parts[0] >= "011") or (parts[1] < "010") or (parts[1] > "022"):
        #     continue

        # Do only Edgeøya
        # if (parts[0] < "016") or (parts[0] > "021") or (parts[1] < "016") or (parts[1] > "021"):
        #     continue

        # if not Path(f"temp.svalbard/medians/svalbard/dem_noncoreg/median_dem_2017_chunks_3584/chunk_{chunk_nr}.tif").is_file() and not redo:
        #     continue
        
        out_paths = get_mosaic_filenames(chunk_nr=chunk_nr, chunksize=chunksize)

        if not all(fp.is_file() for fp in out_paths.values()) or redo:
            call_args.append({"chunk_nr": chunk_nr, "chunksize": chunksize, "redo": redo})
            # continue # TEMPORARY

        all_out_paths.append(out_paths)

    if len(call_args) > 0:
        if n_workers == 1:
            for args in tqdm.tqdm(call_args,smoothing=0.1, desc="Mosaicking and fitting trend.", disable=True):
                process_chunk_wrapper(args, verbose=True)
        else:
            tqdm.contrib.concurrent.process_map(process_chunk_wrapper, call_args, max_workers=n_workers, smoothing=0.1, desc="Mosaicking and fitting trend.")


    per_key_filepaths = {}
    for paths in all_out_paths:

        for key in paths:
            if key not in per_key_filepaths:
                per_key_filepaths[key] = []
            
            per_key_filepaths[key].append(paths[key].absolute())

    out_dir = Path(f"temp.svalbard/filt/svalbard/mosaics_{chunksize}/")

    from osgeo import gdal
    gdal.UseExceptions()

    periods = {
                "0": "2013-2018",
                "1": "2019-2024",
                "full": "2013-2024",
            }


    out_dir.mkdir(exist_ok=True)
    # raise NotImplementedError()

    for key in per_key_filepaths:

        name = key
        if key.startswith("20") or key == "quality_mode":
            name = f"dem_{key}"
        elif "accel" in key:
            name = f"trend_{periods['full']}_{'_'.join(key.split('_')[::-1])}"
        elif (period := key.split("_")[1]) in ["0", "1", "full"]:
            name = f"trend_{periods[period]}_{key.split('_')[0]}"
            

        out_path = out_dir / f"{name}.vrt"
        gdal.BuildVRT(
            str(out_path),
            [str(fp) for fp in filter(lambda fp: fp.is_file(), per_key_filepaths[key])]
        )
      

def sample_point_timeseries_trends():
    workdir = Path("temp.svalbard/filt/svalbard/mosaics_3584")
    raster_ext = ".vrt"
    csv_path = Path("temp.svalbard/point_timeseries_trends.csv")

    points = [
        (569452, 8691793, "Rabotbreen front"),
        (539077, 8605225, "Doktorbreen surge"),
        (667622, 8875455, "Austfonna summit"),
        (448486, 8757319, "Kronebreen terminus retreat"),
        (513015, 8579753, "Amundsenisen"),
        (437100, 8825883, "Lilliehöökbreen accumulation area"),
    ]

    years = list(range(2013, 2025))
    dem_products = ["dem", "count", "doy", "nmad"]
    trend_intervals = ["2013-2018", "2019-2024", "2013-2024"]
    trend_products = ["slope", "intercept", "slope_fit_err", "slope_baseline_err"]

    quality_labels = {
        1: "q95",
        2: "q75",
        3: "bias",
        4: "vertcorr",
    }

    def decode_value(dataset: rio.io.DatasetReader, raw: np.ndarray | float) -> np.ndarray | float:
        scale = dataset.scales[0] if dataset.scales else 1.0
        offset = dataset.offsets[0] if dataset.offsets else 0.0
        nodata = dataset.nodata
        arr = np.asarray(raw, dtype=np.float64)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = arr * scale + offset
        if np.isscalar(raw):
            return float(arr)
        return arr

    def read_point_value(path: Path, x: float, y: float) -> float:
        with rio.open(path) as ds:
            row, col = ds.index(x, y)
            raw = ds.read(1, window=((row, row + 1), (col, col + 1)), masked=False)[0, 0]
            return float(decode_value(ds, raw))

    def sample_point_series(x: float, y: float):
        records = []
        for year in years:
            rec = {"year": year}
            for product in dem_products:
                path = workdir / f"dem_{year}_{product}{raster_ext}"
                rec[product] = read_point_value(path, x, y)
            rec["x"] = year + (rec["doy"] - 1.0) / 365.25
            records.append(rec)
        return records

    def trend_product_name(interval: str, product: str) -> str:
        if interval == "2019-2024" and product == "slope":
            return "slope_tcorr"
        if interval == "2019-2024" and product == "slope_fit_err":
            return "slope_tcorr_fit_err"
        return product

    def sample_trends(x: float, y: float):
        out = {}
        for interval in trend_intervals:
            for product in trend_products:
                raster_product = trend_product_name(interval, product)
                path = workdir / f"trend_{interval}_{raster_product}{raster_ext}"
                out.setdefault(interval, {})[product] = read_point_value(path, x, y)
        return out

    def sample_point_row(x: float, y: float, label: str) -> dict:
        quality_raw = read_point_value(workdir / f"dem_quality_mode{raster_ext}", x, y)
        quality = quality_labels.get(int(round(quality_raw)), f"unknown({quality_raw})")
        series = sample_point_series(x, y)
        trends = sample_trends(x, y)

        row = {
            "label": label,
            "x": x,
            "y": y,
            "quality_mode": quality,
        }
        for year_rec in series:
            year = year_rec["year"]
            row[f"{year}_dem"] = year_rec["dem"]
            row[f"{year}_nmad"] = year_rec["nmad"]
            row[f"{year}_count"] = year_rec["count"]
            row[f"{year}_doy"] = year_rec["doy"]
        for interval in trend_intervals:
            trend = trends[interval]
            row[f"{interval}_slope"] = trend["slope"]
            row[f"{interval}_intercept"] = trend["intercept"]
            row[f"{interval}_fit_err"] = trend["slope_fit_err"]
            row[f"{interval}_baseline_err"] = trend["slope_baseline_err"]
        return row

    rows = [sample_point_row(x, y, label) for x, y, label in points]
    columns = ["label", "x", "y", "quality_mode"]
    for year in years:
        columns.extend([f"{year}_dem", f"{year}_nmad", f"{year}_count", f"{year}_doy"])
    for interval in trend_intervals:
        columns.extend([f"{interval}_slope", f"{interval}_intercept", f"{interval}_fit_err", f"{interval}_baseline_err"])

    pd.DataFrame.from_records(rows).reindex(columns=columns).to_csv(csv_path, index=False, na_rep="")


if __name__ == "__main__":
    mosaic_all_tiles()
