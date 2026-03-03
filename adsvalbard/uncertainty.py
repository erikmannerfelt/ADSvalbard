import rasterio as rio
import pandas as pd
import geopandas as gpd
import numpy as np
import shutil
import tqdm
import tqdm.contrib.concurrent

from pathlib import Path
import matplotlib.pyplot as plt
# import xdem

import adsvalbard.utilities
import adsvalbard.rasters


def get_temp_dir() -> Path:
    return Path("temp.svalbard/uncertainty")

def get_chunks_dir() -> Path:
    return Path("./temp.svalbard/filt/svalbard/chunks_3584")

def var_name_to_se_name(var_col: str) -> str:
    return var_col.replace("_slope", "_se").replace("_accel", "_accel_se") 


def sample_points(n_points: int = 25000, redo: bool = False):
    temp_dir = get_temp_dir()
    out_path = temp_dir / f"sampled_pts_{n_points}.arrow"

    if out_path.is_file() and not redo:
        return gpd.read_feather(out_path)

    base_filepath = "./temp.svalbard/filt/svalbard/mosaics_3584/"
    var_keys = [
        "trend_2013-2024_slope",
        "trend_2013-2018_slope",
        "trend_2019-2024_slope",
        "trend_2013-2024_accel",
    ]
    keys = var_keys.copy()
    for key in keys.copy():
        keys.append(var_name_to_se_name(key))

    files = {key: Path(f"{base_filepath}{key}.vrt") for key in keys}
    files["terr_slope_of_slope"] = Path("temp/npi_slope_of_slope.tif")
    files["terr_slope"] = Path("temp/npi_slope.tif")

    with rio.open(files[var_keys[0]]) as raster:
        bounds = raster.bounds

    points: pd.DataFrame = adsvalbard.utilities.create_random_points(bounds, n_points=n_points)

    for key, fp in reversed(files.items()):
        points[key] = adsvalbard.rasters.sample_raster(fp, points.geometry)
        # print(points.shape[0])
        if key.endswith("_slope") or key.endswith("_accel"):
            points.loc[points[key] == 0., key] = np.nan

        if key == "terr_slope":
            points.drop(points.index[points[key] == 0.], inplace=True)

        if key not in var_keys:
            points.dropna(inplace=True, subset=[key])
        if points.shape[0] == 0:
            raise ValueError("No points left!")

    # points.dropna(inplace=True, subset=[key for key in keys if key not in var_keys])

    # Read stable terrain mask
    points["stable"] = adsvalbard.rasters.sample_raster(Path("temp/stable_terrain.tif"), points.geometry) == 1

    out_path.parent.mkdir(exist_ok=True)
    points.dropna().to_feather(out_path)

    return gpd.read_feather(out_path)



def make_err_function(points: gpd.GeoDataFrame, pred_col: str, var_cols: list[str] = ["terr_slope", "terr_slope_of_slope"]):
    bins = {col: np.unique(np.percentile(points[col], np.arange(0, 100, 10))) for col in var_cols}

    import xdem
    df = xdem.spatialstats.nd_binning(
        values=points.loc[points["stable"], pred_col].values,
        list_var=[points.loc[points["stable"], col].values for col in var_cols],
        list_var_names=var_cols,
        statistics=["count", xdem.spatialstats.nmad],
        list_var_bins=[bins[key] for key in var_cols],
    )
    
    dh_err_func_unscaled = xdem.spatialstats.interp_nd_binning(df, list_var_names=var_cols, statistic="nmad", min_count=5)

    _, dh_err_fun = xdem.spatialstats.two_step_standardization(
        points.loc[points["stable"], pred_col].values,
        list_var=[points.loc[points["stable"], col].values for col in var_cols],
        unscaled_error_fun=dh_err_func_unscaled,
    )

    points["dh_err_pred"] = dh_err_fun(points[var_cols])

    full_nmad = xdem.spatialstats.nmad(points.loc[points["stable"], pred_col])
    pred_nmad = points["dh_err_pred"].median()

    assert np.abs(full_nmad - pred_nmad) < 0.1, f"{full_nmad:.3f}!={pred_nmad:.3f}"

    return dh_err_fun
    

def apply_function(chunk_dir: Path, func, pred_col: str, var_cols: list[str], redo: bool = False):

    temp_dir = get_temp_dir()
    # chunk_dir = get_chunks_dir() / f"chunk_{chunk_nr}"
    chunk_nr = chunk_dir.stem.replace("chunk_", "")

    out_path = temp_dir / f"{chunk_dir.parts[-2]}/chunk_{chunk_nr}_{pred_col}_err_sigma1.tif"

    if out_path.is_file() and not redo:
        return out_path
    se_col = var_name_to_se_name(pred_col)

    # arrs = {}

    # with rio.open(chunk_dir / f"chunk_{chunk_nr}_{pred_col}.tif") as raster:

    #     arrs = pd.DataFrame({pred_col:raster.read(1, masked=True).filled(np.nan).ravel()

    with rio.open(chunk_dir / f"chunk_{chunk_nr}_{se_col}.tif") as raster:
        err = raster.read(1, masked=True)
        err = (err * raster.scales[0] + raster.offsets[0]).filled(np.nan).ravel()
        # err = raster.read(1, masked=True).filled(np.nan).ravel()
        bounds = raster.bounds
        meta = raster.meta


    paths=  {
        "terr_slope": Path("temp/npi_slope.tif"),
        "terr_slope_of_slope": Path("temp/npi_slope_of_slope.tif")
    }

    arrs = pd.DataFrame(index=np.arange(err.size))

    for key in var_cols:
        with rio.open(paths[key]) as raster:
            window = rio.windows.from_bounds(*bounds, raster.transform)
            print((meta["height"], meta["width"]), window)
            arr = raster.read(1, masked=True, window=window, boundless=True)
            arrs[key] = (arr * raster.scales[0] + raster.offsets[0]).filled(np.nan).ravel()

    err = np.hypot(func(arrs), err)

    scale = (65534 / 10)
    err = np.where(
        np.isfinite(err),
        np.clip(err * scale, a_min=0, a_max=65534),
        65535,
    ).astype("uint16")
    temp_path = out_path.with_suffix(".tif.tmp")

    temp_path.parent.mkdir(exist_ok=True, parents=True)
    with rio.open(temp_path, "w", zlevel=9, compress="deflate", tiled=True, **meta) as raster:
        raster.write(err.reshape((meta["height"], meta["width"])), 1)
        raster.scales = (1 / scale,)


    shutil.move(temp_path, out_path)

    
    return out_path

    

def sample_wrapper(kwargs):
    # filepath, points, dem_title = args
    points = kwargs["points"]
    vals = adsvalbard.rasters.sample_raster(filepath=kwargs["filepath"], geometry=points.geometry)

    tag = adsvalbard.rasters.sample_raster(filepath=kwargs["filepath"].with_name(kwargs["filepath"].name.replace("_dem.tif", "_matchtag.tif")), geometry=points.geometry) == 1
    vals[~tag] = np.nan

    return pd.Series(vals, index=pd.MultiIndex.from_arrays((points["id"], [kwargs["dem_title"]] * points.shape[0]), names=["id", "title"]))
    return points

def check_bad_pts(gui: bool = False):

    ylims = {
        "001": [528, 585],
        "002": [402, 621],
        "003": [177, 227],
        "004": [97, 150],
        "005": [415, 531],
        "006": [147, 260],
        "007": [554, 583],
        "008": [546, 579],
        "009": [955, 970],
        "010": [675, 742],
        "011": [373, 453],
        "012": [449, 480],
        "013": [148, 205],
        "014": [283, 347],
        "015": [262, 286],
        "016": [114, 268],
        "017": [684, 737],
        "018": [664, 691],
        "019": [510, 573],
        "020": [517, 540],
        "021": [481, 533],
        "022": [1190, 1209],
        "023": [900, 921],
        "024": [603, 641],
        "025": [399, 508],
        "026": [181, 221],
        "027": [201, 247],
        "028": [183, 227],
        "029": [268, 293],
        "030": [626, 670],
        "031": [933, 973],
        "032": [551, 599],
        "033": [731, 756],
        "034": [344, 374],
        "035": [267, 317],
        "036": [475, 520],
        "037": [371, 388],
        "038": [625, 647],
        "039": [550, 590],
        "040": [317, 345],
        "041": [472, 498],
        "042": [58, 84],
        "043": [203, 236],
        "044": [360, 408],
        "045": [790, 809],
        "046": [416, 446],
        "047": [466, 486],
        "048": [327, 344],
        "049": [232, 286],
        "050": [275, 314],
        "051": [63, 78],
        "052": [313, 345],
        "053": [536, 553],
        "054": [134, 150],
        "055": [204, 255],
    }

    points = gpd.read_file("shapes/bad_vertcoreg_pts.geojson")
    n_points = points.shape[0]

    strips = gpd.read_feather("temp.svalbard/strip-meta.feather")

    joined = gpd.sjoin(points, strips)

    joined["datetime"] = pd.to_datetime(joined["datetime"])
    joined["year"] = joined["datetime"].dt.year + joined["datetime"].dt.month / 12

    titles = []
    years = []
    sample_call_args = []
    for title, grouped in joined.groupby("title"):
        filepath = Path(f"data/ArcticDEM/{title}_dem.tif")


        if not filepath.is_file():
            continue
        years.append(grouped["year"].iloc[0])
        titles.append(title)
        sample_call_args.append(
            {
                "points": grouped[["id", "geometry", "year"]].to_crs(3413).copy(),
                "filepath": filepath,
                "dem_title": title,
            }
        )

    years = pd.Series(years, index=titles)

    # print(sample_wrapper(sample_call_args[1]))
    # return

    points = pd.concat(tqdm.contrib.concurrent.process_map(sample_wrapper, sample_call_args, chunksize=1)).dropna()

    # good_titles = []
    # bad_titles = []
    # for key in ylims:
    #     pts = points.loc[(key, slice(None))]

    #     bad = (pts < ylims[key][0]) | (pts > ylims[key][1])
    #     for title, isbad in bad.items():
    #         if isbad and title not in good_titles and title not in bad_titles:
    #             bad_titles.append(title)

    #         if not isbad:
    #             if title in bad_titles:
    #                 bad_titles.remove(title)
    #             good_titles.append(title)

    from collections import Counter

    good_counts = Counter()   # title -> number of good observations
    bad_titles = set()        # use a set to avoid duplicates

    for key, (ymin, ymax) in ylims.items():
        pts = points.loc[(key, slice(None))]

        bad = (pts < ymin) | (pts > ymax)

        for title, isbad in bad.items():
            if isbad:
                # Only mark bad if we've never (or not yet twice) seen it as good
                if good_counts[title] < 3:
                    bad_titles.add(title)
                # bad_titles.add(title)
            else:
                ...
                # Count good observations
                good_counts[title] += 1
                # If at least two good observations, it's no longer bad
                if good_counts[title] >= 3:
                    bad_titles.discard(title)  # discard: no error if not present

    Path("temp.svalbard/bad_dems.txt").write_text("\n".join(bad_titles))

    if gui:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes: list[plt.Axes] = fig.subplots(4, int(np.ceil(n_points / 4)), sharex=False).ravel().tolist()

        point_labels = []
        for i, (point_id, point_res) in enumerate(points.groupby("id")):

            titles = point_res.index.get_level_values("title")
            vals = pd.Series(point_res.values, years.loc[titles])
            point_labels.append(point_id)

            axis = axes[i]
            axis.set_title(point_id)
            # if point_id in ylims:
            #     axis.set_ylim(*ylims[point_id])
            axis.scatter(vals.index, vals, c=np.isin(titles, list(bad_titles)).astype(int))
        plt.subplots_adjust(bottom=0.2)
        from matplotlib.widgets import Button
        button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        button = Button(button_ax, 'Get y-lims')

        def on_button_clicked(event):
            # Loop over all axes in the figure
            for i, ax in enumerate(fig.axes):
                # Skip non-data axes (like the button axis)
                if ax is button_ax:
                    continue
        
                if i >= len(point_labels):
                    continue
                ylim = [round(v) for v in ax.get_ylim()]
                print(f'"{point_labels[i]}": {ylim},')
        button.on_clicked(on_button_clicked)

        plt.show()
    

def main():

    points = sample_points()

    var_cols = ["terr_slope", "terr_slope_of_slope"]
    pred_cols = [
        "trend_2013-2024_slope",
        "trend_2013-2018_slope",
        "trend_2019-2024_slope",
        "trend_2013-2024_accel",
    ]

    chunks_dir = get_chunks_dir()

    chunk_dirs = list(filter(lambda d: d.is_dir(), chunks_dir.glob("chunk_*_*")))

    for pred_col in pred_cols:
    # pred_col = "trend_2013-2024_slope"
        se_col = var_name_to_se_name(pred_col)
        func = make_err_function(points, pred_col=pred_col, var_cols=var_cols)

        for chunk_dir in tqdm.tqdm(chunk_dirs, desc=f"Applying err functions for {pred_col}"):
            apply_function(chunk_dir, func=func, pred_col=pred_col, var_cols=var_cols)
    return
    points["dh_err_pred"] = np.hypot(func(points[var_cols]), points[se_col]) 

    vmin, vmax = np.nanpercentile(points["dh_err_pred"], [2, 98])
    plt.scatter(points.geometry.x, points.geometry.y, c=points["dh_err_pred"], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
