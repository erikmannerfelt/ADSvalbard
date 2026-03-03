import rasterio as rio
import geopandas as gpd
import pandas as pd
import shapely.geometry
import numpy as np
import scipy.optimize
import warnings
import tqdm
import tqdm.contrib.concurrent
import json

from pathlib import Path

def sample_raster(filepath, points):

    with rio.open(filepath) as raster:
        fill = np.nan if "float" in str(raster.dtypes[0]) else 0

        arr = np.fromiter(
            map(
                lambda v: v[0],#.filled(fill),
                raster.sample(
                    points,
                    # masked=True,
                )
            ),
            dtype=raster.dtypes[0],
            count=points.shape[0]
        )
        # return arr
        return np.where(
            arr != raster.nodata,
            arr,
            fill
        )


def sample_dem(matchtag_fp: Path, sample_pts: np.ndarray) -> dict[str, np.ndarray | None]:
    title = matchtag_fp.stem.split("_matchtag")[0]
    dem_fp = matchtag_fp.with_stem(matchtag_fp.stem.replace("matchtag", "dem"))
    matchtag = sample_raster(matchtag_fp, sample_pts)

    valid = matchtag > 0

    if np.count_nonzero(valid) == 0:
        return {title: None}

    dem = np.full(valid.shape[0], np.nan, dtype="float32")

    dem[valid] = sample_raster(dem_fp, sample_pts[valid])
    return {title: dem}

def sample_dem_wrapper(args):
    return sample_dem(**args)

def coregister_vert_year(year: int, bounds: rio.coords.BoundingBox, label: str | None = None, n_points: int = 1000, verbose: bool = True, redo: bool = False) -> pd.Series:

    if label is not None:
        label = f"_{label}"
    else:
        label = ""

    out_path = Path(f"temp.svalbard/vertcoreg_results/vertcoreg_results_{year}{label}.csv")
    out_meta_path = out_path.with_name(out_path.stem + "_meta.json")

    if all(fp.is_file() for fp in (out_path, out_meta_path)) and not redo:
        return pd.read_csv(out_path, header=None, index_col=0).squeeze()

    bounds_shp = shapely.geometry.box(*bounds)

    all_strips = gpd.read_feather("temp.svalbard/strip-meta.feather")
    all_strips["datetime"] = pd.to_datetime(all_strips["datetime"])
    all_strips = all_strips[all_strips["datetime"].dt.year == year]

    bad_titles = Path("temp.svalbard/bad_dems.txt").read_text().splitlines()
    all_strips = all_strips[~all_strips["title"].isin(bad_titles)]

    all_strips = all_strips[all_strips.intersects(bounds_shp)]

    Path("tmp").mkdir(exist_ok=True)
    all_strips.to_feather(f"tmp/strips_{year}.arrow")

    rng: np.random.Generator = np.random.default_rng(0)
    x_pts = rng.uniform(bounds.left, bounds.right, size=n_points * 3)
    y_pts = rng.uniform(bounds.bottom, bounds.top, size=n_points * 3)
    sample_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_pts, y_pts, crs=all_strips.crs))

    sample_points["x"] = sample_points.geometry.x
    sample_points["y"] = sample_points.geometry.y
    sample_points = sample_points.sort_values(["x", "y"])

    if verbose:
        print("Reading land outlines")
    land = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Land_f.shp", bbox=bounds_shp).simplify(5).to_frame().dissolve().to_crs(32633)

    if verbose:
        print("Reading glacier outlines")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        glaciers = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Isbreer_f.shp", bbox=bounds_shp).simplify(5).to_frame().dissolve().to_crs(32633)


    sample_points = sample_points[sample_points.intersects(land.geometry[0])]
    sample_points = sample_points.iloc[:n_points]

    sample_points["stable"] = ~sample_points.intersects(glaciers.geometry[0])

    sample_arr = np.column_stack((sample_points.geometry.x, sample_points.geometry.y))
    sample_points.loc[sample_points["stable"], "npi_dem"] = sample_raster("temp/npi_mosaic.vrt", sample_arr[sample_points["stable"]])
    def cost(params: np.ndarray,
         arrs: np.ndarray,
         stable_elev: np.ndarray,
         x: np.ndarray,
         y: np.ndarray,
         center_x: np.ndarray,
         center_y: np.ndarray,
         w_stable: float = 3.,
         progress_bar = None) -> np.ndarray:
        """
        params:      (3m,) [a_0, b_0, c_0, a_1, b_1, c_1, ..., a_{m-1}, b_{m-1}, c_{m-1}]
        arrs:        (n, m)  DEM samples, no npi_dem
        stable_elev: (n,)    npi_dem elevations (NaN where not available)
        x, y:        (n,)    point coordinates in same CRS as DEMs
        center_x/y:  (m,)    reference x/y center per DEM (same order as columns)
        w_stable:    weight for the stable terrain residuals
        """
        n_pts, n_dems = arrs.shape
        coeffs = params.reshape(n_dems, 3)  # columns: [a, b, c]

        a = coeffs[:, 0]     # (m,)
        b = coeffs[:, 1]     # (m,)
        c0 = coeffs[:, 2]    # (m,)

        # Broadcasted deltas: (n, 1) - (1, m) -> (n, m)
        dx = x[:, None] - center_x[None, :]
        dy = y[:, None] - center_y[None, :]

        # Planar correction per DEM: (n, m)
        plane_corr = a[None, :] * dx + b[None, :] * dy + c0[None, :]

        arrs_shifted = arrs + plane_corr   # (n, m)

        # 1) DEM–DEM consistency per row
        row_med = np.nanmedian(arrs_shifted, axis=1)            # (n,)
        intra_res = arrs_shifted - row_med[:, None]             # (n, m)
        intra_res = intra_res[np.isfinite(intra_res)]           # 1D

        if progress_bar is not None:
            progress_bar.update()
        # 2) DEM–npi_dem residuals
        stable_mask = np.isfinite(stable_elev)
        if np.any(stable_mask):
            stable_res = arrs_shifted[stable_mask, :] - stable_elev[stable_mask, None]
            stable_res = stable_res[np.isfinite(stable_res)]    # 1D
            stable_res = w_stable * stable_res
            return np.r_[intra_res, stable_res]
        else:
            return intra_res


    # def cost(offsets: np.ndarray,
    #      arrs: np.ndarray,
    #      stable_elev: np.ndarray,
    #      w_stable: float = 0.5) -> np.ndarray:
    #     """
    #     offsets:     (m,)   per-column offsets
    #     arrs:        (n, m)
    #     stable_elev: (n,)   reference elevation at each row (NaN where not available)
    #     w_stable:    weight for the stable terrain residuals
    #     """
    #     # Apply column offsets
    #     arrs_shifted = arrs + offsets[None, :]  # (n, m)

    #     # -----------------------------
    #     # 1) Align datasets per row
    #     # -----------------------------
    #     row_med = np.nanmedian(arrs_shifted, axis=1)       # (n,)
    #     intra_res = arrs_shifted - row_med[:, None]        # (n, m)
    #     intra_res = intra_res[np.isfinite(intra_res)]      # 1D

    #     # return intra_res
    #     # -----------------------------
    #     # 2) Align to stable terrain
    #     # -----------------------------
    #     stable_mask = np.isfinite(stable_elev)
    #     if np.any(stable_mask):
    #         # residuals w.r.t. reference where reference exists
    #         stable_res = arrs_shifted[stable_mask, :] - stable_elev[stable_mask, None]
    #         stable_res = stable_res[np.isfinite(stable_res)]  # 1D
    #         stable_res = w_stable * stable_res
    #         return np.r_[intra_res, stable_res]
    #     else:
    #         # No stable reference available
    #         return intra_res

    centroids = []
    call_args = []
    for _, strip in all_strips.iterrows():
        matchtag_fp = Path("temp.svalbard/arcticdem_vrts/") / (strip["matchtag"].split("/")[-1].split(".")[0] + "_epsg32633_5.0m.vrt")

        if not matchtag_fp.is_file():
            continue

        call_args.append({"matchtag_fp": matchtag_fp, "sample_pts": sample_arr.copy()})

        centroids.append(
            {
                "name": matchtag_fp.stem.split("_matchtag")[0],
                "center_x": strip.geometry.centroid.x,
                "center_y": strip.geometry.centroid.y,
            }
        )

    centroids = pd.DataFrame.from_records(centroids).set_index("name")

    # print(centroids)
    # raise NotImplementedError()

    arrs_raw = tqdm.contrib.concurrent.process_map(sample_dem_wrapper, call_args, desc="Sampling", smoothing=0.1, max_workers=10, disable=(not verbose))
    # arrs_raw = []
    # for args in tqdm.tqdm(call_args):
    #     arrs_raw.append(sample_dem_wrapper(args))

    arrs = {}
    for arr_raw in arrs_raw:
        for key, value in arr_raw.items():
            if value is None:
                continue
            arrs[key] = value

    arrs = pd.DataFrame(arrs, index=sample_points.index)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        med = np.nanmedian(arrs.values, axis=1)
        diff_pre = np.nanmedian((arrs - med[:, None]), axis=0)

    # If they're very far away from the median from the start, something's wrong
    too_much = np.abs(diff_pre) > 75
    arrs = arrs.loc[:, ~too_much]

    # _n = pd.Series(diff_pre, arrs.columns)
    # print(_n.sort_values().iloc[-5:])
    # raise NotImplementedError()

    arrs["npi_dem"] = sample_points["npi_dem"].values
    arrs = arrs.iloc[np.count_nonzero(np.isfinite(arrs), axis=1) > 1].dropna(how="all", axis="columns")

    # Split DEM columns and reference
    dem_cols = [c for c in arrs.columns if c != "npi_dem"]
    dem_vals = arrs[dem_cols].values              # (n, m)
    n_pts, n_dems = dem_vals.shape

    stable_elev = arrs["npi_dem"].values          # (n,)

    # Coordinates for each sample point (already sorted earlier)
    x = sample_points.loc[arrs.index, "x"].values
    y = sample_points.loc[arrs.index, "y"].values

    # ---------- Diagnostics BEFORE ----------
    # use simple row-median NMAD and DEM–npi_dem median, like before
    stable_terrain_before = np.nanmedian(dem_vals - stable_elev[:, None])
    med = np.nanmedian(dem_vals, axis=1)
    nmad_pre = 1.4826 * np.nanmedian(np.abs(dem_vals - med[:, None]))

    if verbose:
        print("Co-registering (planar: a*x + b*y + c per DEM)")

    # Initial parameters: zeros (no correction)
    # x0 = np.zeros(3 * n_dems)
    x0 = np.tile([0.0, 0.0, -28.0], n_dems)
    max_slope = 5.0 / 50000.0

    # lower and upper bounds for [a, b, c] of one DEM
    lb_one = [-max_slope, -max_slope, -np.inf]
    ub_one = [ max_slope,  max_slope,  np.inf]

    # repeat for all DEMs
    lb = np.tile(lb_one, n_dems)
    ub = np.tile(ub_one, n_dems)

    center_x = centroids.loc[dem_cols, "center_x"].values
    center_y = centroids.loc[dem_cols, "center_y"].values

    max_iters = 10000

    # Solve
    with tqdm.tqdm(total=max_iters, desc="Co-registering (max iters:)", disable=(not verbose)) as progress_bar:
        res = scipy.optimize.least_squares(
            cost,
            loss="soft_l1",
            x0=x0,
            max_nfev=max_iters,
            args=(dem_vals, stable_elev, x, y, center_x, center_y),
            kwargs={"progress_bar": progress_bar},
            bounds=(lb, ub),
        )

    params = res.x.reshape(n_dems, 3)   # columns: [a, b, c] per DEM

    # ---------- Diagnostics AFTER ----------
    dx = x[:, None] - center_x[None, :]
    dy = y[:, None] - center_y[None, :]
    a = params[:, 0]
    b = params[:, 1]
    c0 = params[:, 2]

    plane_corr = a[None, :] * dx + b[None, :] * dy + c0[None, :]
    dem_vals_corr = dem_vals + plane_corr

    stable_terrain_after = np.nanmedian(dem_vals_corr - stable_elev[:, None])
    med = np.nanmedian(dem_vals_corr, axis=1)
    nmad_post = 1.4826 * np.nanmedian(np.abs(dem_vals_corr - med[:, None]))

    # ---------- Save coefficients ----------
    # a small DataFrame: index = DEM name, columns = [a, b, c]
    shifts = pd.DataFrame(params, index=dem_cols, columns=["a", "b", "c"])
    shifts["center_x"] = center_x
    shifts["center_y"] = center_y
    shifts["n_comparisons"] = 0
    # Count comparison points like before
    n_cmp_pts = {}
    valids = np.isfinite(arrs.values)
    for i, key in enumerate(arrs.columns):
        if key == "npi_dem":
            continue

        idxs = np.arange(arrs.shape[1])
        idxs = idxs[idxs != i]

        n_cmps = np.count_nonzero(valids[valids[:, i]][:, idxs])
        n_cmp_pts[key] = int(n_cmps)
        shifts.loc[key, "n_comparisons"] = int(n_cmps)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    shifts.to_csv(out_path)

    stats = {
        "n_points": n_points,
        "n_stable_points": int(np.count_nonzero(sample_points["stable"])),
        "stable_terrain_median_pre": float(stable_terrain_before),
        "stable_terrain_median_post": float(stable_terrain_after),
        "nmad_pre": float(nmad_pre),
        "nmad_post": float(nmad_post),
        "n_comparisons": n_cmp_pts,
    }
    out_meta_path.write_text(json.dumps(stats, indent=2))

    # Return a Series of parameters (flattened a,b,c per DEM)
    return shifts

    # arrs = pd.DataFrame(arrs)

    # arrs["npi_dem"] = sample_points["npi_dem"].values
    # arrs = arrs.iloc[np.count_nonzero(np.isfinite(arrs), axis=1) > 1].dropna(how="all", axis="columns")

    # stable_terrain_before = np.nanmedian(arrs.values[:, :-1] - arrs.values[:, [-1]])
    # med = np.nanmedian(arrs.values[:, :-1], axis=1)       # (n,)
    # nmad_pre = 1.4826 * np.nanmedian(np.abs(arrs.values[:, :-1] - med[:, None]))        # (n, m)
    # if verbose:
    #     print("Co-registering")
    # x = scipy.optimize.least_squares(cost, args=(arrs.values[:, :-1],arrs.values[:, -1]), x0=np.zeros(arrs.shape[1] - 1))

    # stable_terrain_after = np.nanmedian(arrs.values[:, :-1] + x.x[None, :] - arrs.values[:, [-1]])
    # med = np.nanmedian(arrs.values[:, :-1] + x.x[None, :], axis=1)       # (n,)
    # nmad_post = 1.4826 * np.nanmedian(np.abs(arrs.values[:, :-1] + x.x[None, :] - med[:, None]))        # (n, m)

    # shifts = pd.Series(x.x, arrs.columns[:-1])
    # out_path.parent.mkdir(exist_ok=True, parents=True)
    # shifts.to_csv(out_path, header=False)

    # n_cmp_pts = {}

    # valids = np.isfinite(arrs.values)
    # for i, key in enumerate(arrs):
    #     if key == "npi_dem":
    #         continue

    #     idxs = np.arange(arrs.shape[1])
    #     idxs = idxs[idxs != i]

    #     n_cmps = np.count_nonzero(valids[valids[:, i]][:,  idxs])
    #     n_cmp_pts[key] = int(n_cmps)

    # stats = {
    #             "n_points": n_points,
    #             "n_stable_points": int(np.count_nonzero(sample_points["stable"])),
    #             "stable_terrain_median_pre": float(stable_terrain_before),
    #             "stable_terrain_median_post": float(stable_terrain_after),
    #             "nmad_pre": float(nmad_pre),
    #             "nmad_post": float(nmad_post),
    #             "n_comparisons": n_cmp_pts,
    #         }
    # out_meta_path.write_text(
    #     json.dumps(
    #         stats,
    #     )
    # )

    # return pd.read_csv(out_path, header=None, index_col=0).squeeze()
    

def main(n_points: int = 5000, verbose: bool = True):

    import adsvalbard.stacking
    bounds = rio.coords.BoundingBox(627722.5, 8784882.5,  753162.5, 8946162.5)

    for year in range(2013, 2025):
        # if year == 2015:
        #     continue
        print(year)
        points = coregister_vert_year(year, label="austfonna", bounds=bounds, n_points=n_points, verbose=verbose, redo=True)
        # raise NotImplementedError()

        # print(points)
        # continue

        # return
        # adsvalbard.stacking.create_median_stack(years=year, n_threads=5, raster_type="dem_vertcoreg", bounds_override=bounds, vertcoreg_label="austfonna", nchunks_override=1)

    # import adsvalbard.mosaicking
    # adsvalbard.mosaicking.mosaic_tile("005_018")

    # new_chunk_nrs = set()
    # for year in range(2013, 2025):
    #     print(f"Vertcoreg: {year}")
    #     for raster_type in ["dem", "dem_vertcoreg", "dem_noncoreg"]:
    #         new_paths = adsvalbard.stacking.create_median_stack(years=year, n_threads=5 if year <= 2014 or year >= 2022 else 2, raster_type=raster_type, bounds_override=bounds, vertcoreg_label="austfonna", nchunks_override=None, outlier_threshold=0.75)

    #     # for chunk_nr in map(lambda fp: fp.stem.replace("chunk_", ""), new_paths):
        #     new_chunk_nrs.add(chunk_nr)


    # import shutil
    # for chunk_nr in new_chunk_nrs:
    #   dir_path = Path("/home/erik/Projects/storage/UiO/ADSvalbard/temp.svalbard/filt/svalbard/chunks_3584/") / f"chunk_{chunk_nr}"

    #   shutil.move(dir_path, dir_path / f"../chunks_3584.bkp/chunk_{chunk_nr}")
    #   print(dir_path.is_dir())
    #   # print(chunk_nr)

    # return

    new_paths = adsvalbard.stacking.create_median_stack(years=2021, n_threads=1,raster_type="dem_vertcoreg", vertcoreg_label="austfonna", outlier_threshold=0.95, bounds_override=bounds)
    import os

    chunk_nrs = [fp.stem for fp in new_paths]

    for demdir in ["filt/svalbard/", "medians/svalbard/dem_noncoreg", "medians/svalbard/dem_vertcoreg"]:
        for filepath in Path(f"temp.svalbard/{demdir}/").rglob("*chunk_*.tif"):
            if filepath.stem not in chunk_nrs:
                continue
            os.remove(filepath)
            

    # return
    # import os
    # for chunk_nr in new_chunk_nrs:
    #     paths = adsvalbard.mosaicking.mosaic_tile(chunk_nr)

    #     with rio.open(paths["2015_quality"]) as raster:

    #         uniques, counts = np.unique(raster.read(1), return_counts=True)

    #     if 4 not in uniques:
    #         print(f"Chunk {chunk_nr} not ready")# {pd.Series(counts, uniques)}")
    #         os.remove(paths["2015_quality"])
    #     else:
    #         print(f"Chunk {chunk_nr} compatible")

    #       # print(pd.Series(counts, uniques))

    #   # print(paths)
    #   # return
    for year in range(2013, 2025):
        for raster_type in ["dem", "dem_noncoreg", "dem_vertcoreg"]:
            print(f"{raster_type}: {year}")
            new_paths = adsvalbard.stacking.create_median_stack(years=year, n_threads=5 if year <= 2014 or year >= 2022 else 2, raster_type=raster_type, vertcoreg_label="austfonna", outlier_threshold=0.95, bounds_override=bounds if raster_type == "dem_vertcoreg" else None)

            # print(new_paths)
            # return

    import adsvalbard.mosaicking
    adsvalbard.mosaicking.mosaic_tile("006_021", redo=True)
    adsvalbard.mosaicking.mosaic_all_tiles()

    return
        

    import adsvalbard.mosaicking
    return

    for year in range(2013, 2025):
        print(year)
        points = coregister_vert_year(year, label="austfonna", bounds=bounds, n_points=n_points, verbose=verbose, redo=False)

        # print(points)

        # return
        import adsvalbard.stacking
        adsvalbard.stacking.create_median_stack(years=year, n_threads=2, raster_type="dem_vertcoreg", bounds_override=bounds, vertcoreg_label="austfonna", nchunks_override=None)
        


if __name__ == "__main__":
    main()

    
