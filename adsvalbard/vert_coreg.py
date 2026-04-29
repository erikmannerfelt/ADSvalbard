from adsvalbard.mosaicking import write_formatted
from adsvalbard.outlines import generate_full_glacier_mask
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
import tempfile

from pathlib import Path
from adsvalbard.constants import CONSTANTS

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

def coregister_vert_year(year: int, bounds: rio.coords.BoundingBox, label: str | None = None, n_points: int = 300, verbose: bool = True, redo: bool = False,max_slope = 10.0 / 50000.0) -> pd.Series:
    

    if label is not None:
        label = f"_{label}"
    else:
        label = ""

    out_path = Path(f"temp.svalbard/vertcoreg_results/vertcoreg_results_{year}{label}.csv")
    out_meta_path = out_path.with_name(out_path.stem + "_meta.json")

    if all(fp.is_file() for fp in (out_path, out_meta_path)) and not redo:
        return pd.read_csv(out_path, index_col=0)

    bounds_shp = shapely.geometry.box(*bounds)

    all_strips = gpd.read_feather("temp.svalbard/strip-meta.feather")
    all_strips["datetime"] = pd.to_datetime(all_strips["datetime"])
    all_strips = all_strips[all_strips["datetime"].dt.year == year]

    bad_titles = Path("temp.svalbard/bad_dems.txt").read_text().splitlines()
    bad_titles = list(set(bad_titles + Path("temp.svalbard/bad_dems_dem_vertcoreg.txt").read_text().splitlines()))
    all_strips = all_strips[~all_strips["title"].isin(bad_titles)]

    all_strips = all_strips[all_strips.intersects(bounds_shp)]

    Path("tmp").mkdir(exist_ok=True)
    all_strips.to_feather(f"tmp/strips_{year}.arrow")

    rng: np.random.Generator = np.random.default_rng(0)
    x_pts = rng.uniform(bounds.left, bounds.right, size=n_points * 3)
    y_pts = rng.uniform(bounds.bottom, bounds.top, size=n_points * 3)
    sample_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_pts, y_pts, crs=all_strips.crs))


    if verbose:
        print("Reading land outlines")
    land = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Land_f.shp", bbox=bounds_shp).simplify(5).to_frame().dissolve().to_crs(32633)

    if verbose:
        print("Reading glacier outlines")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        glaciers = gpd.read_file("zip://data/outlines/NP_S100_SHP.zip/NP_S100_SHP/S100_Isbreer_f.shp", bbox=bounds_shp).simplify(5).to_frame().dissolve().to_crs(32633)


    sample_points = sample_points[sample_points.intersects(land.geometry[0])]

    sample_points["stable"] = ~sample_points.intersects(glaciers.geometry[0])

    extra_sample_points = gpd.read_file("shapes/extra_vertcoreg_sample_points.geojson", bbox=bounds_shp)
    extra_sample_points["stable"] = True
    sample_points = pd.concat([extra_sample_points, sample_points], ignore_index=True)

    sample_points = sample_points.iloc[:n_points]

    sample_points["x"] = sample_points.geometry.x
    sample_points["y"] = sample_points.geometry.y
    # import matplotlib.pyplot as plt
    # plt.scatter(sample_points.geometry.x, sample_points.geometry.y, c=sample_points["stable"])
    # plt.show()
    # raise NotImplemented

    sample_points = sample_points.sort_values(["x", "y"])

    sample_points.to_feather(f"tmp/vertcoreg_sample_pts_{year}_{label}.arrow")
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

    # lower and upper bounds for [a, b, c] of one DEM
    lb_one = [-max_slope, -max_slope, -np.inf]
    ub_one = [ max_slope,  max_slope,  np.inf]

    # repeat for all DEMs
    lb = np.tile(lb_one, n_dems)
    ub = np.tile(ub_one, n_dems)

    center_x = centroids.loc[dem_cols, "center_x"].values
    center_y = centroids.loc[dem_cols, "center_y"].values

    max_iters = 10000
    # from multiprocessing import Pool

    tol = 1e-7

    # from IPython import embed
    # embed()

    # Solve
    with tqdm.tqdm(total=max_iters, desc="Co-registering (max iters:)", disable=(not verbose)) as progress_bar:
        res = scipy.optimize.least_squares(
            cost,
            loss="soft_l1",
            x0=x0,
            ftol=tol,
            gtol=tol,
            xtol=tol,
            # x_scale="jac",
            # workers=Pool.map,
            max_nfev=max_iters,
            args=(dem_vals, stable_elev, x, y, center_x, center_y),
            kwargs={"progress_bar": progress_bar},
            bounds=(lb, ub),
        )
    # res = noisyopt.minimizeCompass(
    #     cost,
    #     x0=x0,
    #     args=(dem_vals, stable_elev, x, y, center_x, center_y),
    #     bounds=np.column_stack((lb, ub)),
    #     disp=True,
    #     paired=False,
    # )
    # res = noisyopt.minimizeSPSA(
    #     cost,
    #     x0=x0,
    #     args=(dem_vals, stable_elev, x, y, center_x, center_y),
    #     bounds=np.column_stack((lb, ub)),
    #     disp=True,
    #     niter=1000,
    #     paired=False,
    # )

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
        "n_points": sample_points.shape[0],
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
    

def get_vertcoreg_region_bounds() -> dict[str, "rio.coords.BoundingBox"]:
    chunk_outlines = gpd.read_file(CONSTANTS.temp_dir.with_stem("temp.svalbard") / "chunk_outlines.geojson")
    bad_regions = gpd.read_file("shapes/bad_coreg_regions.geojson")

    bounds = {}
    for _, region in bad_regions.iterrows():
        intersecting = chunk_outlines.intersects(region.geometry)

        bounds[region["label"]] = rio.coords.BoundingBox(
            *chunk_outlines.loc[
                chunk_outlines.touches(
                    chunk_outlines.loc[intersecting].dissolve().iloc[0]["geometry"]
                )
            ].total_bounds
        )
    
    return bounds
    # return {
    #     "austfonna": rio.coords.BoundingBox(627722.5, 8800002.5, 760002.5, 8946162.5),
    # }[label]



def main(n_points: int = 5000, verbose: bool = True):

    import adsvalbard.stacking

    vertcoreg_region_bounds = get_vertcoreg_region_bounds()

    for label, bounds in vertcoreg_region_bounds.items():
        print(label)
        for year in range(2013, 2025):
            # if year == 2015:
            #     continue
            print(year)
            kwargs = {}
            # There's so little stable terrain that deramp is ill posed
            if label == "kvitoya":
                kwargs["max_slope"] = 1e-6
            # print("NOTE: REDOING KVITOYA")
            points = coregister_vert_year(year, label=label, bounds=bounds, n_points=n_points, verbose=verbose, redo=False, **kwargs)

            # raise NotImplementedError()

            # print(points)
            # continue

            # return
            # adsvalbard.stacking.create_median_stack(years=year, n_threads=5, raster_type="dem_vertcoreg", bounds_override=bounds, vertcoreg_label="austfonna", nchunks_override=1)

    # return
    # return
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

    # new_paths = adsvalbard.stacking.create_median_stack(years=2015, n_threads=2,raster_type="dem_noncoreg", vertcoreg_label="austfonna", outlier_threshold=0.95, bounds_override=bounds)
    # chunk_nrs = [fp.stem for fp in new_paths]
    # # return
    # uniques, counts = np.unique(([fp.stem.replace("_count", "") for fp in Path("temp.svalbard/medians/svalbard/dem_vertcoreg/").rglob("*/chunk_*count.tif")]), return_counts=True)
    # missing_vertcoreg_chunks = uniques[counts != 12].tolist() + [nr for nr in chunk_nrs if nr not in uniques]


    # print([nr for nr in chunk_nrs if nr not in existing_vertcoreg_chunks])

    # return


    # for demdir in ["filt/svalbard/chunks_3584/"]:#, "medians/svalbard/dem_vertcoreg"]:
    #     for filepath in Path(f"temp.svalbard/{demdir}/").rglob("*chunk_*.tif"):
    #         if not any(nr in filepath.stem for nr in missing_vertcoreg_chunks):
    #             continue
    #         # print(filepath)
    #         os.remove(filepath)
    # return
            

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
    for label, bounds in vertcoreg_region_bounds.items():
        print(label)
        for year in range(2013, 2025):
            for raster_type in ["dem", "dem_noncoreg", "dem_vertcoreg"]:
                print(f"{raster_type}: {year}")
                filt = [0.75, 0.95] if raster_type == "dem" else [0.75]

                for threshold in filt:
                    new_paths = adsvalbard.stacking.create_median_stack(
                        years=year,
                        n_threads=5 if year <= 2014 or year >= 2022 else 2,
                        # n_threads=1,
                        raster_type=raster_type,
                        vertcoreg_label=label,
                        outlier_threshold=threshold,
                        bounds_override=bounds if raster_type == "dem_vertcoreg" else None)

            # print(new_paths)
            # return

    import adsvalbard.mosaicking
    # adsvalbard.mosaicking.mosaic_tile("005_021", redo=True)
    adsvalbard.mosaicking.mosaic_all_tiles(redo=False)
    # temporal_biascorr(redo=True)

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
        

def temporal_biascorr_inner(chunks: gpd.GeoDataFrame, product_str: str, mosaic_dir: Path, year: int, redo: bool = False, max_vcorr_magnitude: float = 0.6, return_first_chunk: bool = False, filepath_override: dict[str, Path] = {}, verbose: bool = True):
    import rasterio as rio
    import matplotlib.pyplot as plt
    import scipy.ndimage
    import scipy.spatial
    import scipy.interpolate
    import adsvalbard.rasters
    import adsvalbard.mosaicking
    import shapely.geometry

    filepaths = {
        "ocean": mosaic_dir / f"dem_{year}_ocean.vrt",
        "stable_terrain": CONSTANTS.temp_dir / "stable_terrain.tif",
        "quality": mosaic_dir / f"dem_{year}_quality.vrt",
        "nmad": mosaic_dir / f"{product_str}_nmad.vrt",
        "slope": mosaic_dir / f"{product_str}_slope.vrt",
    } | filepath_override

    res = CONSTANTS.res

    bounds = rio.coords.BoundingBox(*chunks.total_bounds)
    with rio.open(filepaths["ocean"]) as raster:
        window = rio.windows.from_bounds(*bounds, raster.transform)
        if verbose:
            print("Reading ocean")
        ocean = raster.read(1, window=window, boundless=True) == 1

        arr_bounds = rio.windows.bounds(window, raster.transform)
        if verbose:
            print("Constructing coords")
        x_coords, y_coords = np.meshgrid(
            np.linspace(arr_bounds[0] + raster.res[0] / 2, arr_bounds[2] - raster.res[0] / 2, ocean.shape[1]), 
            np.linspace(arr_bounds[1] + raster.res[1] / 2, arr_bounds[3] - raster.res[1] / 2, ocean.shape[0])[::-1], 
        )

        window_transform = rio.windows.transform(window, raster.transform)
    with rio.open(filepaths["stable_terrain"]) as raster:
        if verbose:
            print("Reading stable")
        window = rio.windows.from_bounds(*bounds, raster.transform)
        stable_terrain = raster.read(1, window=window, boundless=True) == 1
        
    glaciers = (~ocean) & (~stable_terrain)

    with rio.open(filepaths["quality"]) as raster:
        if verbose:
            print("Reading quality")
        window = rio.windows.from_bounds(*bounds, raster.transform)
        missing_data = (raster.read(1, window=window, boundless=True) == 0)

    if verbose:
        print("Dilating")
    patch = missing_data & glaciers
    valid = (~missing_data) & glaciers

    edge_invalid = patch & scipy.ndimage.binary_dilation(valid, structure=np.ones((3, 3)))
    edge_valid   = valid & scipy.ndimage.binary_dilation(patch, structure=np.ones((3, 3)))

    if verbose:
        print("Constructing coords")
    pts = pd.DataFrame({"x_valid": x_coords[edge_valid], "y_valid": y_coords[edge_valid]})
    invalid_pts = pd.DataFrame({"x_invalid": x_coords[edge_invalid], "y_invalid": y_coords[edge_invalid]})


    tree = scipy.spatial.cKDTree(invalid_pts[["x_invalid", "y_invalid"]])

    distances, indices = tree.query(pts[["x_valid", "y_valid"]], k=1)

    distance_mask = distances < res * 2
    pts = pts[distance_mask]

    for axis in ["x", "y"]:
        pts[f"{axis}_invalid"] = invalid_pts.loc[indices, f"{axis}_invalid"].values[distance_mask]
        pts[axis] = pts[[f"{axis}_valid", f"{axis}_invalid"]].mean(axis="columns")
    

    # plt.scatter(pts["x"], pts["y"])
    # plt.show()

    # with rio.open(mosaic_dir / "trend_2019-2024_slope.vrt") as raster:
    #     window = rio.windows.from_bounds(*bounds, raster.transform)

    #     vals = (raster.read(1, masked=True, boundless=True, window=window).astype("float32") * raster.scales[0]).filled(np.nan)
    #
    #
    #

    if verbose:
        print("Sampling NMAD")
    smp = adsvalbard.rasters.sample_raster(
        filepaths["nmad"],
        gpd.points_from_xy(
            x=np.r_[pts["x_valid"], pts["x_invalid"]],
            y=np.r_[pts["y_valid"], pts["y_invalid"]],
        )
    )
    if verbose:
        print(pts.shape)
    pts = pts[np.max((smp[:pts.shape[0]], smp[pts.shape[0]:]), axis=0) < 2.]

    if verbose:
        print("Sampling slope")
    smp = adsvalbard.rasters.sample_raster(
        filepaths["slope"],
        gpd.points_from_xy(
            x=np.r_[pts["x_valid"], pts["x_invalid"]],
            y=np.r_[pts["y_valid"], pts["y_invalid"]],
        )
    )
    if verbose:
        print(pts.shape)
    pts["val_valid"] = smp[:pts.shape[0]]
    pts["val_invalid"] = smp[pts.shape[0]:]
    pts["diff"] = pts["val_valid"] - pts["val_invalid"]
    pts = pts[np.abs(pts["diff"]) < max_vcorr_magnitude]
    if verbose:
        print(pts.shape)
    # pts["diff"] = 1.
    # pts = pts.dropna(subset="diff")

    bin_size = 1000
    xbins = np.arange(arr_bounds[0], arr_bounds[2] + bin_size, bin_size)
    ybins = np.arange(arr_bounds[1], arr_bounds[3] + bin_size, bin_size)
    pts["y_bin"] = np.digitize(pts["y"], ybins)
    pts["x_bin"] = np.digitize(pts["x"], xbins)

    grid = []
    for _, subset in pts.groupby("x_bin"):
        grouped = subset.select_dtypes(np.number).groupby("y_bin")
        new = grouped.median()
        new["count"] = grouped["diff"].count()
        grid.append(new)

    if len(grid) == 0:
        raise ValueError(f"Grid empty.\nPts: {pts}")
    grid = pd.concat(grid)

    if verbose:
        print("Interpolating diff")
    model = scipy.interpolate.RBFInterpolator(grid[["x", "y"]], grid["diff"].astype("float32"), kernel="linear")


    for _, chunk in chunks.iterrows():

        if chunk["tbias_filepath"].is_file() and not redo:
            print(f"Skipping {chunk['tbias_filepath'].name} as it already exists")
            continue
        if verbose:
            print(chunk["chunk_id"])
        chunk_bounds = chunk.geometry.bounds
        window = rio.windows.from_bounds(*chunk_bounds, window_transform)
        # for attr in ["col_off", "row_off", "height", "width"]:
        #     setattr(window, attr, round(getattr(window, attr)))

        # chunk_height = int((chunk_bounds[3] - chunk_bounds[1]) / res)
        # chunk_width = int((chunk_bounds[2] - chunk_bounds[0]) / res)
        x_coords, y_coords = np.meshgrid(
            np.linspace(chunk_bounds[0] + res / 2, chunk_bounds[2] - res / 2, int(window.width)), 
            np.linspace(chunk_bounds[1] + res / 2, chunk_bounds[3] - res / 2, int(window.height))[::-1], 
        )

        pred = np.zeros((int(window.height), int(window.width)), dtype="float32")


        patch_sub = patch[*window.toslices()]
        # with rio.open(mosaic_dir / "dem_2024_quality.vrt") as raster:
        #     print("Reading quality")
        #     window = rio.windows.from_bounds(*chunk_bounds, raster.transform)
        #     missing_2024_sub = (raster.read(1, window=window, boundless=True) == 0)
        


        pred[patch_sub] = model(
            np.column_stack(
                (
                    x_coords[patch_sub].ravel(),
                    y_coords[patch_sub].ravel(),
                )
            )
        )

        if return_first_chunk:
            return pred

        with rio.open(chunk["slope_filepath"]) as raster:
            raster_params = raster.meta
            # trend = (raster.read(1, masked=True).astype("float32") * raster.scales[0]).filled(np.nan)
        adsvalbard.mosaicking.write_formatted(
            "slope_",  # This will trigger the same write profile as the slope rasters
            chunk["tbias_filepath"],
            pred,
            params=raster_params,
            redo=True,
        )

        
        # continue
        # plt.imshow(pred, cmap="RdBu", vmin=-0.5, vmax=0.5)
        # fig = plt.figure()
        # axes = fig.subplots(1, 3, sharex=True, sharey=True)
        # axes[0].imshow(trend, cmap="RdBu", vmin=-1, vmax=1)
        # axes[1].imshow(pred, cmap="RdBu", vmin=-1, vmax=1)
        # axes[2].imshow(trend + pred, cmap="RdBu", vmin=-1, vmax=1)
        # plt.show()

    # xx, yy = np.meshgrid(xbins, ybins)
    # # print(pred)
    # fig = plt.figure()
    # axes = fig.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(pred.reshape(xx.shape), vmin=-0.5, vmax=0.5, cmap="RdBu", extent=(0, missing_2024.shape[1], 0, missing_2024.shape[0]))
    # axes[1].imshow(missing_2024)
    # plt.show()

    # plt.scatter(xx.ravel(), yy.ravel(), c=pred, cmap="RdBu", vmin=-.3, vmax=.3)
    # corr = np.zeros(missing_2024.shape, dtype="float32")
    # corr[missing_2024] = model(np.column_stack((x_coords[missing_2024], y_coords[missing_2024])))

    # plt.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
    # plt.colorbar()
    # plt.show()

    # fig = plt.figure()
    # axes = fig.subplots(1, 3, sharex=True, sharey=True)
    # axes[0].imshow(edge_invalid)
    # axes[1].imshow(glaciers & missing_2024)
    # axes[2].imshow(edge_valid)
    # plt.show()
    


def create_summed_vrt(path1: Path, path2: Path, vrt_path: Path, band=1):
    from osgeo import gdal
    import os
    gdal.UseExceptions()
    ds1 = gdal.Open(path1)
    ds2 = gdal.Open(path2)

    if ds1 is None or ds2 is None:
        raise RuntimeError("Could not open one or both input rasters.")

    # Basic compatibility checks
    if (ds1.RasterXSize != ds2.RasterXSize or
        ds1.RasterYSize != ds2.RasterYSize):
        raise ValueError("Raster dimensions do not match.")

    if ds1.GetGeoTransform() != ds2.GetGeoTransform():
        raise ValueError("GeoTransforms do not match.")

    if ds1.GetProjection() != ds2.GetProjection():
        raise ValueError("Projections do not match.")

    b1 = ds1.GetRasterBand(band)
    b2 = ds2.GetRasterBand(band)

    scale1 = b1.GetScale()
    offset1 = b1.GetOffset()
    scale2 = b2.GetScale()
    offset2 = b2.GetOffset()

    # Normalize None values
    scale1 = 1.0 if scale1 is None else scale1
    scale2 = 1.0 if scale2 is None else scale2
    offset1 = 0.0 if offset1 is None else offset1
    offset2 = 0.0 if offset2 is None else offset2

    if scale1 != scale2 or offset1 != offset2:
        raise ValueError(
            f"Input scale/offset differ: "
            f"({scale1}, {offset1}) vs ({scale2}, {offset2})"
        )

    dtype_name = gdal.GetDataTypeName(b1.DataType)
    nodata = b1.GetNoDataValue()

    gt = ds1.GetGeoTransform()
    proj = ds1.GetProjection()

    nodata_xml = f"<NoDataValue>{nodata}</NoDataValue>" if nodata is not None else ""

    vrt_xml = f'''<VRTDataset rasterXSize="{ds1.RasterXSize}" rasterYSize="{ds1.RasterYSize}">
      <SRS>{proj}</SRS>
      <GeoTransform>{", ".join(map(str, gt))}</GeoTransform>
      <VRTRasterBand dataType="{dtype_name}" band="1" subClass="VRTDerivedRasterBand">
        <ColorInterp>Gray</ColorInterp>
        <PixelFunctionType>sum</PixelFunctionType>
        {nodata_xml}
        <SimpleSource>
          <SourceFilename relativeToVRT="0">{os.path.abspath(path1)}</SourceFilename>
          <SourceBand>{band}</SourceBand>
        </SimpleSource>
        <SimpleSource>
          <SourceFilename relativeToVRT="0">{os.path.abspath(path2)}</SourceFilename>
          <SourceBand>{band}</SourceBand>
        </SimpleSource>
      </VRTRasterBand>
    </VRTDataset>
    '''

    with open(vrt_path, "w", encoding="utf-8") as f:
        f.write(vrt_xml)

    # Reopen in update mode and set output scale/offset metadata
    out_ds = gdal.Open(vrt_path, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetScale(scale1)
    out_band.SetOffset(offset1)
    out_band = None
    out_ds = None

    ds1 = None
    ds2 = None

    # A hack to fix the rounding that GDAL does above. It rounds to 17 decimals instead of 19 which make the VRTs
    # incompatible...
    with open(vrt_path) as infile:
        vrt_content = infile.read()
    vrt_content = vrt_content.replace(f"{scale1:.17f}", str(scale1))
    with open(vrt_path, "w") as outfile:
        outfile.write(vrt_content)

    
def get_temporal_biascorr_fixes() -> list[tuple[float, tuple[float, float, float, float]]]:
    fixes = [
        (2024, (693517, 8823582,730963, 8930063)), # E Austfonna
        (2024, (659308,8894594,672531,8918846)), # Duvebreen
        # (2024, (604638.5465,8861780.8961,634862.2933,8905775.9468)), # Vestfonna
        # (458480.3002,8771056.3376,465609.6921,8800975.4095), # Isachsenfonna
        (2024, (458850.2849,8771734.1000,460606.0061,8789243.8596)), # Isachsenfonna narrow
        (2019, (609672.0295,8887533.0667,627532.6505,8905734.2504)), # Vestfonna 2
    ]
    return fixes
    

def temporal_biascorr(redo: bool = False):
    import rasterio as rio
    import matplotlib.pyplot as plt
    import shapely.geometry
    from osgeo import gdal
    gdal.UseExceptions()

    temp_dir = CONSTANTS.temp_dir.with_stem("temp.svalbard")
    mosaic_dir = temp_dir / "filt/svalbard/mosaics_3584"
    chunk_dir = mosaic_dir.with_stem(mosaic_dir.stem.replace("mosaics", "chunks"))

    product_str = "trend_2019-2024"

    all_chunks = gpd.read_file(temp_dir / "chunk_outlines.geojson")

    fixes = get_temporal_biascorr_fixes()
    
    all_chunks["tbias_filepath"] = all_chunks["chunk_id"].apply(lambda chunk_id: chunk_dir / f"{chunk_id}/{chunk_id}_{product_str}_slope_tbias.tif")
    all_chunks["slope_filepath"] = all_chunks["tbias_filepath"].apply(lambda fp: fp.with_name(fp.name.replace("_tbias.tif", ".tif")))
    all_chunks["tcorr_filepath"] = all_chunks["tbias_filepath"].apply(lambda fp: fp.with_name(fp.name.replace("_tbias.tif", "_tcorr.vrt")))


    for i, (year, bounds) in enumerate(fixes):
        print(f"Fix {i+1} / {len(fixes)}: {rio.coords.BoundingBox(*bounds)}")
        chunks = all_chunks[all_chunks.intersects(shapely.geometry.box(*bounds))]


        # old_redo = redo
        # if year == 2019:
        #     redo = True
        if not chunks["tbias_filepath"].apply(lambda fp: fp.is_file()).all() or redo:
            kwargs = {"max_vcorr_magnitude": 1.} if bounds[0] == 458850.2849 else {}  # Isachsenfonna
            temporal_biascorr_inner(chunks=chunks, product_str=product_str, redo=redo, mosaic_dir=mosaic_dir, year=year, **kwargs)

        # redo = old_redo

        for _, chunk in tqdm.tqdm(chunks.iterrows(), total=chunks.shape[0]):

            # vrt_filepath: Path = chunk["filepath"].with_name(chunk["filepath"].name.replace("_tbias.tif", "_tcorr.vrt"))
            # slope_filepath: Path = chunk["filepath"].with_name(chunk["filepath"].name.replace("_tbias.tif", ".tif"))

            with rio.open(chunk["slope_filepath"]) as raster:
                if raster.dtypes[0] == "float32":
                    import adsvalbard.mosaicking
                    adsvalbard.mosaicking.mosaic_tile(chunk_nr=chunk["chunk_id"].replace("chunk_", ""), redo=True)

            # print(chunk["slope_filepath"])
            create_summed_vrt(chunk["slope_filepath"], chunk["tbias_filepath"], chunk["tcorr_filepath"])

    in_filepaths = []
    for _, chunk in all_chunks.iterrows():
        if not chunk["slope_filepath"].is_file():
            continue

        # There are some poorly formatted tiles left that should be removed
        with rio.open(chunk["slope_filepath"]) as raster:
            dtype = raster.dtypes[0]
        if dtype == "float32":
            import os
            os.remove(chunk["slope_filepath"])
            continue
        if chunk["tcorr_filepath"].is_file():
            in_filepaths.append(str(chunk["tcorr_filepath"]))
        else:
            in_filepaths.append(str(chunk["slope_filepath"]))

    gdal.BuildVRT(
        mosaic_dir / f"{product_str}_slope_tcorr.vrt",
        in_filepaths,
    )
   


if __name__ == "__main__":
    main()

    
