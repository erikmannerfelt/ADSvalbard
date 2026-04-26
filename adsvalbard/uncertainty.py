from adsvalbard import outlines, vert_coreg
from adsvalbard.constants import CONSTANTS
import rasterio as rio
import pandas as pd
import geopandas as gpd
import numpy as np
import shutil
import tqdm
import tqdm.contrib.concurrent
import warnings

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


def sample_points(n_points: int = int(1e6), redo: bool = False):
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
        "dem_quality_mode",
    ]
    keys = var_keys.copy()
    for key in keys.copy():
        keys.append(var_name_to_se_name(key))

    files = {key: Path(f"{base_filepath}{key}.vrt") for key in keys}
    files["terr_curvature"] = Path("temp/npi_curvature.vrt")
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

    points["easting"] = points.geometry.x
    points["northing"] = points.geometry.y

    out_path.parent.mkdir(exist_ok=True)
    points.dropna().to_feather(out_path)

    return gpd.read_feather(out_path)



def make_err_function(points: gpd.GeoDataFrame, pred_col: str, var_cols: list[str]):

    bins = {}
    for col in var_cols:
        if col == "dem_quality_mode":
            bins["dem_quality_mode"] = [0, 3, 5]
        elif col in ["easting", "northing"]:
            bins[col] = np.unique(np.percentile(points[col], np.linspace(0, 100, 5)))
        else:
            bins[col] = np.unique(np.percentile(points[col], np.linspace(0, 100, 10)))

    # print(np.unique(points["dem_quality_mode"].astype(int), return_counts=True))
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

    out_path = temp_dir / f"{chunk_dir.parts[-2]}/chunk_{chunk_nr}_{pred_col}_err.tif"

    if out_path.is_file() and not redo:
        return out_path
    se_col = var_name_to_se_name(pred_col)

    # arrs = {}

    # with rio.open(chunk_dir / f"chunk_{chunk_nr}_{pred_col}.tif") as raster:

    #     arrs = pd.DataFrame({pred_col:raster.read(1, masked=True).filled(np.nan).ravel()

    with rio.open(chunk_dir / f"chunk_{chunk_nr}_{se_col}.tif") as raster:
        err = raster.read(1, masked=True)
        err = (err.astype("float32") * raster.scales[0] + raster.offsets[0]).filled(np.nan).ravel()
        # err = raster.read(1, masked=True).filled(np.nan).ravel()
        bounds = raster.bounds
        meta = raster.meta

        eastings, northings = np.meshgrid(
            np.linspace(bounds[0] + raster.res[0] / 2, bounds[2] - raster.res[0] / 2, raster.width), 
            np.linspace(bounds[1] + raster.res[1] / 2, bounds[3] - raster.res[1] / 2, raster.height)[::-1], 
        )

        


    paths=  {
        "terr_slope": Path("temp/npi_slope.tif"),
        "terr_curvature": Path("temp/npi_curvature.vrt"),
        "terr_slope_of_slope": Path("temp/npi_slope_of_slope.tif")
    }

    arrs = pd.DataFrame(index=np.arange(raster.height * raster.width))

    for key in var_cols:
        if key == "easting":
            arrs[key] = eastings.ravel()
        elif key == "northing":
            arrs[key] = northings.ravel()
        else:
            with rio.open(paths[key]) as raster:
                window = rio.windows.from_bounds(*bounds, raster.transform)
                # print((meta["height"], meta["width"]), window)
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

    if kwargs.get("apply_matchtag", False):
        tag = adsvalbard.rasters.sample_raster(filepath=kwargs["filepath"].with_name(kwargs["filepath"].name.replace("_dem", "_matchtag")), geometry=points.geometry) == 1
        vals[~tag] = np.nan
    if (coreg := kwargs.get("vertcoreg_results", None)) is not None:
        corr = coreg["a"] * (points.geometry.x - coreg["center_x"]) + coreg["b"] * (points.geometry.y - coreg["center_y"]) + coreg["c"]
        vals += corr

    if (outlier_threshold := kwargs.get("outlier_threshold", None)) is not None:
        # outlier_proba_path = Path("temp.svalbard/out / f"{year}/{path.stem}_outlier_proba.tif"

        outliers = adsvalbard.rasters.sample_raster(filepath=kwargs["outlier_proba_filepath"], geometry=points.geometry) > (outlier_threshold * 255)
        vals[outliers] = np.nan


    return pd.Series(vals, index=pd.MultiIndex.from_arrays((points["id"], [kwargs["dem_title"]] * points.shape[0]), names=["id", "title"]))

def check_bad_pts(prefix: str = "", gui: bool = False):

    all_ylims = {
        "": {  # This means un-coregistered
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
			"017": [695, 726],
			"018": [664, 691],
			"019": [510, 573],
			"020": [517, 540],
			"021": [481, 533],
			"022": [1190, 1209],
			"023": [900, 921],
			"024": [603, 641],
			"025": [423, 460],
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
			"054": [134, 145],
			"055": [204, 255],
			"056": [544, 621],
			"057": [404, 451],
			"058": [542, 575],
			"059": [587, 637],
			"060": [157, 201],
			"061": [189, 228],
			"062": [395, 411],
			"063": [584, 618],
			"064": [586, 607],
			"065": [610, 648],
			"066": [466, 482],
			"067": [437, 454],
			"068": [426, 493],
			"069": [771, 864],
			"071": [679, 688],
			"072": [716, 730],
			"073": [389, 414],
			"074": [198, 286],
			"075": [623, 644],
			"076": [911, 927],
			"077": [563, 589],
			"078": [217, 304],
			"079": [325, 372],
			"080": [249, 267],
			"081": [326, 367],
        },
        "dem_vertcoreg_old": {
            "001": [557, 568],
            "003": [200, 204],
            "005": [458, 476],
            "007": [569, 573],
            "008": [546, 580],
            "020": [529, 533],
            "024": [615, 628],
            "025": [420, 460],
            "027": [211, 231],
            "028": [181, 227],
            "029": [260, 293],
            "038": [628, 639],
            "039": [563, 573],
            "041": [482, 491],
            "042": [58, 74],
            "045": [790, 809],
            "046": [420, 443],
            "049": [238, 270],
            "050": [288, 310],
            "051": [65, 81],
            "052": [317, 339],
            "053": [537, 553],
            "055": [232, 237],
            "056": [597, 605],
            "058": [557, 563],
            "059": [604, 628],
            "060": [180, 184],
            "082": [429, 471],
        },
        "dem_vertcoreg": {
			"001": [516, 578],
			"003": [192, 214],
			"005": [456, 477],
			"008": [554, 574],
			"025": [417, 460],
			"029": [264, 293],
			"038": [619, 645],
			"039": [549, 601],
			"040": [299, 409],
			"041": [479, 494],
			"042": [50, 88],
			"046": [423, 439],
			"049": [232, 285],
			"052": [322, 339],
			"055": [213, 257],
			"056": [457, 682],
			"058": [538, 608],
			"064": [565, 625],
			"065": [614, 647],
			"082": [429, 476],
			"083": [704, 743],
			"085": [319, 337],
			"086": [149, 172],
			"087": [519, 557],
			"089": [162, 166],
			"090": [771, 785],
			# "091": [518, 528],
        },
        "dem": {  # This means co-registered DEMs
			"011": [376, 381],
			"012": [429, 434],
			"013": [136, 164],
			"014": [271, 298],
			"015": [235, 251],
			"016": [139, 152],
			"017": [633, 716],
			"019": [510, 516],
			"048": [306, 328],
			"069": [769, 770],
			"071": [651, 652],
			"079": [315, 324],
			"080": [223, 238],

        }
    }
    if prefix not in all_ylims:
        print(f"Prefix {prefix} not encountered before")
    ylims = all_ylims.get(prefix, {})

    points = gpd.read_file("shapes/bad_vertcoreg_pts.geojson")
    n_points = points.shape[0]

    strips = gpd.read_feather("temp.svalbard/strip-meta.feather")

    joined = gpd.sjoin(points, strips)

    joined["datetime"] = pd.to_datetime(joined["datetime"])
    joined["year"] = joined["datetime"].dt.year + joined["datetime"].dt.month / 12

    vertcoreg_results = []
    for filepath in Path("temp.svalbard/vertcoreg_results/").glob("*.csv"):
        new = pd.read_csv(filepath, index_col=0)
        vertcoreg_results.append(new)

    vertcoreg_results = pd.concat(vertcoreg_results)

    crs = strips.crs
    apply_matchtag = True
    if prefix == "":
        filepath_pattern = lambda title: Path(f"data/ArcticDEM/{title}_dem.tif")
        crs = 3413
    elif prefix == "dem_vertcoreg":
        filepath_pattern = lambda title: Path(f"temp.svalbard/arcticdem_vrts/{title}_dem_epsg32633.vrt")

    elif prefix == "dem":  # This means co-registered
        filepath_pattern = lambda title: Path(f"temp.svalbard/arcticdem_coreg/{title}_dem_coreg.tif")
        apply_matchtag = False
    # elif prefix == "dem_noncoreg":
    #     filepath_pattern = lambda title: Path(f"temp.svalbard/arcticdem_vrts/{title}_dem_epsg32633.vrt")

    #     import adsvalbard.mosaicking
    #     biases = {}
    #     for year in range(2013, 2025):
    #         biases[year] = adsvalbard.mosaicking.get_uncorr_bias(year)
            
    else:
        raise NotImplementedError()

    titles = []
    years = []
    sample_call_args = []
    skip_pts = []
    for title, grouped in joined.groupby("title"):
        grouped = grouped[~grouped["id"].isin(skip_pts)]
        if grouped.shape[0] == 0:
            continue
        if prefix == "dem" and title in vertcoreg_results.index:
            skip_pts += grouped["id"].unique().tolist()
            continue
        filepath = filepath_pattern(title)
        if not filepath.is_file():
            continue
        years.append(grouped["year"].iloc[0])
        titles.append(title)
        sample_call_args.append(
            {
                "points": grouped[["id", "geometry", "year"]].to_crs(crs).copy(),
                "filepath": filepath,
                "dem_title": title,
                "apply_matchtag": apply_matchtag,
            }
        )

        if prefix == "dem_vertcoreg":
            try:
                sample_call_args[-1]["vertcoreg"] = vertcoreg_results.loc[title].to_dict()
            except KeyError:
                sample_call_args = sample_call_args[:-1]
        elif prefix == "dem":
            sample_call_args[-1]["outlier_proba_filepath"] = Path(f"temp.svalbard/outlier_proba/{grouped['datetime'].iloc[0].year}/{filepath.stem}_outlier_proba.tif")
            sample_call_args[-1]["outlier_threshold"] = 0.75
        # elif prefix == "dem_noncoreg":
        #     sample_call_args[-1]["bias"] = biases[grouped["datetime"].iloc[0].year]
            
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
                if prefix == "dem_vertcoreg":
                    bad_titles.add(title)
                # Only mark bad if we've never (or not yet twice) seen it as good
                elif good_counts[title] < 3:
                    bad_titles.add(title)
                # bad_titles.add(title)
            else:
                ...
                # Count good observations
                good_counts[title] += 1
                # If at least two good observations, it's no longer bad
                if good_counts[title] >= 3 and prefix != "dem_vertcoreg":
                    bad_titles.discard(title)  # discard: no error if not present

    if len(prefix) > 0:
        prefix = f"_{prefix}"
    Path(f"temp.svalbard/bad_dems{prefix}.txt").write_text("\n".join(sorted(bad_titles)))

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
            if point_id in ylims:
                axis.set_ylim(*ylims[point_id])
            axis.scatter(vals.index, vals, c=np.isin(titles, list(bad_titles)).astype(int))
        plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.4)
        from matplotlib.widgets import Button
        button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        button = Button(button_ax, 'Get y-lims')

        initial_ylims = []
        for axis in fig.axes:
            initial_ylims.append([round(v) for v in axis.get_ylim()])

        def on_button_clicked(event):
            # Loop over all axes in the figure
            for i, ax in enumerate(fig.axes):
                # Skip non-data axes (like the button axis)
                if ax is button_ax:
                    continue
        
                if i >= len(point_labels):
                    continue
                ylim = [round(v) for v in ax.get_ylim()]

                if point_labels[i] not in ylims and ylim[0] == initial_ylims[i][0] and ylim[1] == initial_ylims[i][1]:
                    continue
                print(f'\t\t\t"{point_labels[i]}": {ylim},')
        button.on_clicked(on_button_clicked)


        plt.show()
    

def _pad_bounds(bounds: rio.coords.BoundingBox, pad: float) -> rio.coords.BoundingBox:
    return rio.coords.BoundingBox(
        left=bounds.left - pad,
        bottom=bounds.bottom - pad,
        right=bounds.right + pad,
        top=bounds.top + pad,
    )


def _window_from_bounds(bounds: rio.coords.BoundingBox, transform) -> rio.windows.Window:
    return rio.windows.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, transform=transform
    ).round_offsets().round_lengths()


def calculate_max_abs_curvature_chunk(
    dem_path: Path,
    out_path: Path,
    bounds: rio.coords.BoundingBox,
    res: float,
) -> Path:
    pad = res  # 1-pixel halo

    if out_path.is_file():
        return out_path

    import xdem
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(dem_path) as src:
        work_bounds = _pad_bounds(bounds, pad)
        work_window = rio.windows.from_bounds(*work_bounds, src.transform).round_offsets().round_lengths()
        final_window = rio.windows.from_bounds(*bounds, src.transform).round_offsets().round_lengths()

        dem = src.read(1, window=work_window, boundless=True, masked=True).filled(np.nan)

        planc, profc = xdem.terrain.get_terrain_attribute(
            dem=dem,
            attribute=["planform_curvature", "profile_curvature"],
            resolution=res,
        )
        maxc = np.max([np.abs(planc), np.abs(profc)], axis=0)

        row0 = int(final_window.row_off - work_window.row_off)
        col0 = int(final_window.col_off - work_window.col_off)
        height = int(final_window.height)
        width = int(final_window.width)

        maxc = maxc[row0 : row0 + height, col0 : col0 + width]

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            nodata=-9999,
            transform=rio.windows.transform(final_window, src.transform),
            compress="deflate",
            predictor=3,
            zlevel=9
        )

        temp_path = out_path.with_suffix(".tif.tmp")
        with rio.open(temp_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(maxc, nan=-9999.), 1)
        shutil.move(temp_path, out_path)

    return out_path


def calculate_all_max_abs_curvature_chunks() -> Path:
    import adsvalbard.stacking
    from osgeo import gdal

    mosaic_path = Path("temp/npi_curvature.vrt")
    res = CONSTANTS.res
    dem_path = Path("temp/npi_mosaic.vrt")
    chunk_gdf = adsvalbard.stacking.make_chunk_polygons()
    out_paths: list[Path] = []

    for row in tqdm.tqdm(chunk_gdf.itertuples(index=False), total=chunk_gdf.shape[0]):
        bounds = rio.coords.BoundingBox(*row.geometry.bounds)
        out_path = Path(f"temp/curvature/{row.chunk_id}_curvature.tif")
        calculate_max_abs_curvature_chunk(
            dem_path=dem_path,
            out_path=mosaic_path,
            bounds=bounds,
            res=res,
        )
        out_paths.append(out_path)

    gdal.UseExceptions()
    gdal.BuildVRT(str(mosaic_path), list(map(str, out_paths)))

    return mosaic_path


def main(redo: bool = False):
    from osgeo import gdal
    gdal.UseExceptions()

    temp_dir = get_temp_dir()
    points = sample_points()

    var_cols = ["terr_slope", "terr_curvature", "easting", "northing"]#, "dem_quality_mode"]
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
        out_paths = []

        for chunk_dir in tqdm.tqdm(chunk_dirs, desc=f"Applying err functions for {pred_col}"):
            try:
                out_path = apply_function(chunk_dir, func=func, pred_col=pred_col, var_cols=var_cols, redo=redo)
                out_paths.append(str(out_path.absolute()))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error on {chunk_dir}: {e}")

        gdal.BuildVRT(
            str(Path("temp.svalbard/filt/svalbard/mosaics_3584") / f"{pred_col}_err.vrt"),
            out_paths,
        )

    return
    points["dh_err_pred"] = np.hypot(func(points[var_cols]), points[se_col]) 

    vmin, vmax = np.nanpercentile(points["dh_err_pred"], [2, 98])
    plt.scatter(points.geometry.x, points.geometry.y, c=points["dh_err_pred"], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


def patch_method_uncertainty():

    import xdem
    with rio.open("temp/stable_terrain.tif", overview_level=2) as raster:
        bounds = raster.bounds
        stable = raster.read(1, masked=True).filled(0) == 1

    for key in ["trend_2013-2018_slope", "trend_2019-2024_slope", "trend_2013-2024_slope"]:
        trend_path = Path(f"temp.svalbard/filt/svalbard/mosaics_3584/out/{key}.tif")


        scale = rio.open(trend_path).scales[0]
        with rio.open(trend_path, overview_level=2) as raster:
            window = rio.windows.from_bounds(*bounds, raster.transform)

            dhdt = (raster.read(1, masked=True, boundless=True, window=window).astype("float32") * scale).filled(np.nan)

        dhdt = np.where(stable, dhdt, np.nan)

        # import matplotlib.pyplot as plt
        # plt.imshow(dhdt, vmin=-.1, vmax=.1, cmap="RdBu")
        # plt.show()

        areas = 10 ** np.linspace(np.log10(raster.res[0] ** 2), np.log10(5e8), 15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            patches = xdem.spatialstats.patches_method(dhdt, areas=areas, gsd=raster.res[0], vectorized=True).dropna(how="any")

        patches = patches.drop_duplicates(subset=["exact_areas"])
        for col in ["nb_indep_patches", "exact_areas", "areas"]:
            patches[col] = patches[col].round().astype(int)

        patches.to_csv(f"temp.svalbard/uncertainty/patch_method_{key}.csv", index=False)

        print(patches)

    

        

    return
        


if __name__ == "__main__":
    main()
