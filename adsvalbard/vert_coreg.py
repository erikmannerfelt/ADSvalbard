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

    all_strips = all_strips[all_strips.intersects(bounds_shp)]

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

    def cost(offsets: np.ndarray,
         arrs: np.ndarray,
         stable_elev: np.ndarray,
         w_stable: float = 0.5) -> np.ndarray:
        """
        offsets:     (m,)   per-column offsets
        arrs:        (n, m)
        stable_elev: (n,)   reference elevation at each row (NaN where not available)
        w_stable:    weight for the stable terrain residuals
        """
        # Apply column offsets
        arrs_shifted = arrs + offsets[None, :]  # (n, m)

        # -----------------------------
        # 1) Align datasets per row
        # -----------------------------
        row_med = np.nanmedian(arrs_shifted, axis=1)       # (n,)
        intra_res = arrs_shifted - row_med[:, None]        # (n, m)
        intra_res = intra_res[np.isfinite(intra_res)]      # 1D

        # return intra_res
        # -----------------------------
        # 2) Align to stable terrain
        # -----------------------------
        stable_mask = np.isfinite(stable_elev)
        if np.any(stable_mask):
            # residuals w.r.t. reference where reference exists
            stable_res = arrs_shifted[stable_mask, :] - stable_elev[stable_mask, None]
            stable_res = stable_res[np.isfinite(stable_res)]  # 1D
            stable_res = w_stable * stable_res
            return np.r_[intra_res, stable_res]
        else:
            # No stable reference available
            return intra_res

    call_args = []
    for _, strip in all_strips.iterrows():
        matchtag_fp = Path("temp.svalbard/arcticdem_vrts/") / (strip["matchtag"].split("/")[-1].split(".")[0] + "_epsg32633_5.0m.vrt")

        call_args.append({"matchtag_fp": matchtag_fp, "sample_pts": sample_arr.copy()})

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

    arrs = pd.DataFrame(arrs)

    arrs["npi_dem"] = sample_points["npi_dem"].values
    arrs = arrs.iloc[np.count_nonzero(np.isfinite(arrs), axis=1) > 1].dropna(how="all", axis="columns")

    stable_terrain_before = np.nanmedian(arrs.values[:, :-1] - arrs.values[:, [-1]])
    med = np.nanmedian(arrs.values[:, :-1], axis=1)       # (n,)
    nmad_pre = 1.4826 * np.nanmedian(np.abs(arrs.values[:, :-1] - med[:, None]))        # (n, m)
    if verbose:
        print("Co-registering")
    x = scipy.optimize.least_squares(cost, args=(arrs.values[:, :-1],arrs.values[:, -1]), x0=np.zeros(arrs.shape[1] - 1))

    stable_terrain_after = np.nanmedian(arrs.values[:, :-1] + x.x[None, :] - arrs.values[:, [-1]])
    med = np.nanmedian(arrs.values[:, :-1] + x.x[None, :], axis=1)       # (n,)
    nmad_post = 1.4826 * np.nanmedian(np.abs(arrs.values[:, :-1] + x.x[None, :] - med[:, None]))        # (n, m)

    shifts = pd.Series(x.x, arrs.columns[:-1])
    out_path.parent.mkdir(exist_ok=True, parents=True)
    shifts.to_csv(out_path, header=False)

    n_cmp_pts = {}

    valids = np.isfinite(arrs.values)
    for i, key in enumerate(arrs):
        if key == "npi_dem":
            continue

        idxs = np.arange(arrs.shape[1])
        idxs = idxs[idxs != i]

        n_cmps = np.count_nonzero(valids[valids[:, i]][:,  idxs])
        n_cmp_pts[key] = int(n_cmps)

    stats = {
                "n_points": n_points,
                "n_stable_points": int(np.count_nonzero(sample_points["stable"])),
                "stable_terrain_median_pre": float(stable_terrain_before),
                "stable_terrain_median_post": float(stable_terrain_after),
                "nmad_pre": float(nmad_pre),
                "nmad_post": float(nmad_post),
                "n_comparisons": n_cmp_pts,
            }
    out_meta_path.write_text(
        json.dumps(
            stats,
        )
    )

    return pd.read_csv(out_path, header=None, index_col=0).squeeze()
    

def main(n_points: int = 50, verbose: bool = True):

    bounds = rio.coords.BoundingBox(627722.5, 8784882.5,  753162.5, 8946162.5)
    points = coregister_vert_year(2024, label="austfonna", bounds=bounds, n_points=n_points, verbose=verbose, redo=True)

    print(points)

    return
    import adsvalbard.stacking
    adsvalbard.stacking.create_median_stack(years=2024, n_threads=1, raster_type="dem_vertcoreg", bounds_override=bounds, vertcoreg_label="austfonna")
    print(points.to_dict())



if __name__ == "__main__":
    main()

    
