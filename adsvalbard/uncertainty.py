import tempfile
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
import shapely.geometry

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
    df.to_csv(f"temp.svalbard/uncertainty/binned_terrain_err_{pred_col}.csv")
    
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

    tcorr_err_path = out_path.with_stem(out_path.stem.replace("_err", "_tcorr_err"))

    #if tcorr_err_path.is_file():
    #    redo = True
    #    print(tcorr_err_path)

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

    if tcorr_err_path.is_file():

        with rio.open(tcorr_err_path) as raster:
            tcorr_err = (raster.read(1, masked=True).astype("float32") * raster.scales[0]).filled(0).ravel()

        err = np.hypot(tcorr_err, err)
        # if "004_015" in tcorr_err_path.stem:
        #     import matplotlib.pyplot as plt
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(err.copy().reshape((meta["height"], meta["width"])), vmin=0, vmax=0.5)
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(tcorr_err.copy().reshape((meta["height"], meta["width"])), vmin=0, vmax=0.5)
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(err.reshape((meta["height"], meta["width"])), vmin=0, vmax=0.5)
        #     plt.show()

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
        
import numpy as np
import scipy.ndimage


def mix_trends_with_square_gaps(
    trend_with_2024: np.ndarray,
    trend_without_2024: np.ndarray,
    gap_size_px: int,
    *,
    res: float = 1.0,
    edge_buffer_px: int = 30,
    inner_buffer_px: int = 30,
    smallest_gap_px: int = 100,
    max_small_gaps: int = 100,
    min_large_gaps: int = 4,
):
    """
    Mix two trend rasters by inserting square gaps from trend_without_2024 into
    trend_with_2024.

    Parameters
    ----------
    trend_with_2024 : np.ndarray
        Reference trend raster (used outside simulated gaps).
    trend_without_2024 : np.ndarray
        Trend raster to insert inside simulated gaps.
    gap_size_px : int
        Side length of each square gap in pixels.
    res : float, default=1.0
        Pixel size. The returned distance mask is in these units.
    edge_buffer_px : int, default=30
        Width of protected outer border that is always taken from trend_with_2024.
    inner_buffer_px : int, default=30
        Minimum spacing between gap members.
    smallest_gap_px : int, default=100
        Reference size used for scaling the number of gaps.
    max_small_gaps : int, default=100
        Target number of gaps when gap_size_px == smallest_gap_px.
    min_large_gaps : int, default=4
        Minimum target number of gaps for large patches, if they fit.

    Returns
    -------
    mixed : np.ndarray
        Mixed raster.
    dist_mask : np.ndarray
        Distance-to-nearest non-gap pixel, in `res` units.
        Outside gaps, the value is 0.
    gap_mask : np.ndarray
        Boolean mask of simulated gaps.
    """

    if trend_with_2024.shape != trend_without_2024.shape:
        raise ValueError("Input arrays must have the same shape.")

    if trend_with_2024.ndim != 2:
        raise ValueError("Only 2D arrays are supported.")

    if gap_size_px <= 0:
        raise ValueError("gap_size_px must be > 0.")

    h, w = trend_with_2024.shape

    avail_h = h - 2 * edge_buffer_px
    avail_w = w - 2 * edge_buffer_px

    if gap_size_px > avail_h or gap_size_px > avail_w:
        raise ValueError(
            "gap_size_px is too large to fit inside the protected edge buffer."
        )

    # Maximum number of squares that can fit while respecting minimum spacing.
    max_rows = 1 + (avail_h - gap_size_px) // (gap_size_px + inner_buffer_px)
    max_cols = 1 + (avail_w - gap_size_px) // (gap_size_px + inner_buffer_px)

    if max_rows < 1 or max_cols < 1:
        raise ValueError("No gaps can fit with the current settings.")

    max_possible = int(max_rows * max_cols)

    # Scale gap count roughly ~ 1 / gap_area, but never below min_large_gaps if possible.
    target_n = int(round(max_small_gaps * (smallest_gap_px / gap_size_px) ** 2))
    target_n = max(min_large_gaps, target_n)
    target_n = min(max_possible, target_n)

    def choose_grid_shape(target, max_r, max_c, aspect):
        best = None
        for nr in range(1, max_r + 1):
            for nc in range(1, max_c + 1):
                n = nr * nc
                if n > target:
                    continue
                shape_aspect = nc / nr
                score = (
                    target - n,                 # prefer as many as possible up to target
                    abs(np.log(shape_aspect / aspect)),  # prefer chunk-like layout
                )
                if best is None or score < best[0]:
                    best = (score, nr, nc)

        if best is None:
            # Fallback: smallest possible grid
            return 1, 1

        return best[1], best[2]

    nr, nc = choose_grid_shape(target_n, max_rows, max_cols, avail_w / avail_h)

    # Place gaps evenly inside the valid interior.
    y0_min = edge_buffer_px
    y0_max = h - edge_buffer_px - gap_size_px
    x0_min = edge_buffer_px
    x0_max = w - edge_buffer_px - gap_size_px

    if nr == 1:
        y_starts = np.array([(y0_min + y0_max) // 2], dtype=int)
    else:
        y_starts = np.round(np.linspace(y0_min, y0_max, nr)).astype(int)

    if nc == 1:
        x_starts = np.array([(x0_min + x0_max) // 2], dtype=int)
    else:
        x_starts = np.round(np.linspace(x0_min, x0_max, nc)).astype(int)

    gap_mask = np.zeros((h, w), dtype=bool)
    gap_id_mask = np.zeros((h, w), dtype=np.int32)
    
    gap_id = 1
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + gap_size_px
            x1 = x0 + gap_size_px

            gap_mask[y0:y1, x0:x1] = True
            gap_id_mask[y0:y1, x0:x1] = gap_id

            gap_id += 1

    mixed = np.array(trend_with_2024, copy=True)
    mixed[gap_mask] = trend_without_2024[gap_mask]

    # Distance inside gaps to nearest non-gap pixel, in units of `res`.
    dist_mask = scipy.ndimage.distance_transform_edt(gap_mask) * res

    return mixed, dist_mask, gap_mask, gap_id_mask

def get_nmad(data):
    median = np.nanmedian(data, axis=-1)
  
    return 1.4826 * np.nanmedian(np.abs(data - median), axis=-1)


def get_temporal_biascorr_uncertainty_relationship(chunk_nr: str = "005_018", redo: bool = False):
    out_path = Path("temp.svalbard/uncertainty/temporal_biascorr_area_nmad.csv")
    out_path2 = Path("temp.svalbard/uncertainty/temporal_biascorr_distance_nmad.csv")

    if out_path.is_file() and out_path2.is_file() and not redo:
        out = pd.read_csv(out_path).set_index("area_m2").squeeze()
        out2 = pd.read_csv(out_path2).set_index("distance_m").squeeze()
        return (out, out2)
    import adsvalbard.mosaicking
    import adsvalbard.vert_coreg
    base_pattern = f"temp.svalbard/filt/svalbard/chunks_3584/chunk_{chunk_nr}/chunk_{chunk_nr}"
    meta = {}

    chunks = gpd.read_file("temp.svalbard/chunk_outlines.geojson").query(f"chunk_id == 'chunk_{chunk_nr}'")
    chunks["tbias_filepath"] = Path("/dev/null")

    dems = []
    flags = []
    for year in range(2019, 2025):
        if year == 2024:
            dems.append(np.full_like(dems[-1], np.nan))
            flags.append(np.zeros_like(flags[-1]))
            continue
        with rio.open(f"{base_pattern}_{year}.tif") as raster:
            dems.append((raster.read(1, masked=True) * raster.scales[0]).filled(np.nan).ravel()[:, None])
        with rio.open(f"{base_pattern}_{year}_quality.tif") as raster:
            flags.append(raster.read(1).ravel()[:, None])

    dems = np.hstack(dems)
    flags = np.hstack(flags)
            

    with rio.open(f"{base_pattern}_trend_2019-2024_slope.tif") as raster:
        meta = raster.meta
        res = raster.res[0]
        trend_with_2024 = (raster.read(1, masked=True) * raster.scales[0]).filled(np.nan)

        
    trend_without_2024 = adsvalbard.mosaicking.fit_trend(
        dems,
        flags,
        verbose=False,
    )["slope"].reshape(trend_with_2024.shape)


    areas = np.empty((0,))
    area_nmads = np.empty((0,))
    diffs = np.empty((0,))
    distances = np.empty((0,))
    for gap_size_px in tqdm.tqdm(np.linspace(100, 1500, 60, dtype=int)):
        mixed, dist_mask, gap_mask, gap_id_mask =  mix_trends_with_square_gaps(trend_with_2024=trend_with_2024, trend_without_2024=trend_without_2024, gap_size_px=gap_size_px, res=res, min_large_gaps=10)


        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            slope_temp_path = temp_dir / "slope.tif"
            qual_temp_path = temp_dir / "quality.tif"

            with rio.open(slope_temp_path, "w", count=1, height=mixed.shape[0], width=mixed.shape[1], transform=meta["transform"], crs=meta["crs"], dtype="float32") as raster:
                raster.write(mixed, 1) 

            quality = np.full(mixed.shape, 4, dtype="uint8")
            quality[gap_mask] = 0
            with rio.open(qual_temp_path, "w", count=1, height=mixed.shape[0], width=mixed.shape[1], transform=meta["transform"], crs=meta["crs"], dtype="uint8") as raster:
                raster.write(quality, 1) 


            bias: np.ndarray = adsvalbard.vert_coreg.temporal_biascorr_inner(
                chunks=chunks,
                product_str="trend_2019-2024",
                mosaic_dir=Path("temp.svalbard/filt/svalbard/mosaics_3584/"),
                year=2024,
                redo=True,
                return_first_chunk=True,
                filepath_override={
                    "slope": slope_temp_path,
                    "quality": qual_temp_path,
                },
                verbose=False,
            )
        diff_vals = (mixed[gap_mask] + bias[gap_mask]) - trend_with_2024[gap_mask]

        diff_means = []
        for idx in np.unique(gap_id_mask):
            if idx == 0:
                continue

            diff_means.append(np.nanmean(diff_vals[gap_id_mask[gap_mask] == idx]))

        area_nmads = np.append(area_nmads, [get_nmad(diff_means)])
        areas = np.append(areas, [(gap_size_px * res) ** 2])

        diffs = np.concatenate([diffs, diff_vals])
        distances = np.concatenate([distances, dist_mask[gap_mask]])


    

        # import matplotlib.pyplot as plt
        # plt.subplot(1, 4, 1)
        # plt.imshow(mixed, vmin=-1, vmax=1, cmap="RdBu")
        # plt.subplot(1, 4, 2)
        # plt.imshow(dist_mask, vmin=0, vmax=5000)
        # plt.subplot(1, 4, 3)
        # plt.imshow(gap_mask)
        # plt.subplot(1, 4, 4)
        # plt.imshow(bias, vmin=-.1, vmax=.1, cmap="RdBu")

        # plt.show()
    
    out = pd.Series(area_nmads, index=areas, name="nmad")
    out.index.name = "area_m2"
    out.to_csv(out_path)

    dist_bins = np.linspace(-0.01, distances.max() + 0.01, 50)
    bin_centers = (dist_bins[1:] + dist_bins[:-1]) / 2
    digitized = np.digitize(distances, bins=dist_bins)

    nmads = []
    for idx in np.unique(digitized):
        vals = diffs[digitized == idx]

        nmad = get_nmad(vals)
        nmads.append(nmad)

    out2 = pd.Series(nmads, index=bin_centers, name="nmad")
    out2.index.name = "distance_m"
    out2.to_csv(out_path2)

    return out, out2
    import matplotlib.pyplot as plt
    plt.scatter(areas, area_nmads)
    plt.show()
    return

def peak_with_linear_tail(x, params):
    """
    Model: Gaussian peak + linear tail, blended with a logistic switch.

    params = [A, mu, sigma, m, c, k, x0]
    """
    A, mu, sigma, m, c, k, x0 = params

    # Gaussian peak
    g = A * np.exp(- (x - mu)**2 / (2.0 * sigma**2))

    # Linear tail
    ell = m * x + c

    # Logistic switching function
    s = 1.0 / (1.0 + np.exp(k * (x - x0)))

    # Blend: s(x)*g(x) + (1-s(x))*ell(x)
    return g * s + ell * (1.0 - s)


def get_temporal_biascorr_uncertainty(verbose: bool = True, include_neff_standardization: bool = False):
    """

    Parameters
    ----------
    include_neff_standardization
        Standardize each patch so that the integrated error (using a neff model) is equal to the empirical estimation.
        From testing, this changes very little but introduces a lot of complexity. Therefore, it's off by default.
    """
    import adsvalbard.vert_coreg
    import adsvalbard.mosaicking
    import scipy.optimize
    import scipy.ndimage
    import matplotlib.pyplot as plt

    per_area, per_distance = get_temporal_biascorr_uncertainty_relationship(redo=False)

    # Some strange outliers make fitting difficult. This removes them
    per_distance = per_distance.loc[:2800]
    # distance_err_model = lambda distance: spherical_model(distance, distance_model_res.x)
    distance_err_model = np.poly1d(np.polyfit(per_distance.index, per_distance, deg=1))

    per_area.index /= 1e6
    per_area.index.name = "area_km2"


    if include_neff_standardization:
        # This fits a Gaussian peak at around 12 km² (assessed visually) and then it tapers to a linear
        # reducing trend
        area_model_res = scipy.optimize.least_squares(
            lambda params, x, y: peak_with_linear_tail(x, params) - y,
            x0=[
                0.02,   # A      ~ peak height
                12.0,   # mu     ~ peak position
                2.3,    # sigma  ~ peak width
                -0.0002,# m      ~ slope of tail (negative small)
                0.02,   # c      ~ intercept so tail near peak ~ 0.02
                0.5,    # k      ~ transition sharpness
                17.0    # x0     ~ transition location (12 + 5)
            ],
            args=(per_area.index, per_area.values),
        )
        # The area error model is clamped to the lowest trailing values to avoid going <= 0
        area_err_model = lambda area: np.clip(peak_with_linear_tail(area, area_model_res.x), a_min=np.percentile(per_area.iloc[:-10], 10), a_max=np.inf)
        neff_model = lambda area: (area_err_model(area) / area_err_model(0)) ** 2

        neff_data = pd.read_csv("vgm_neff_cirq.csv")
        neff_model = scipy.interpolate.interp1d(neff_data.values[:, 0] / 1e6, neff_data.values[:, 1])
    # plt.scatter(per_distance.index, per_distance)
    # plt.plot(per_distance.index, distance_err_model(per_distance.index))
    # plt.show()

    
    # return

    # xs = np.linspace(0, 100)
    # plt.scatter(per_area.index, per_area)
    # plt.plot(xs, area_err_model(xs))
    # plt.show()
    # return

    all_chunks = gpd.read_file("temp.svalbard/chunk_outlines.geojson")

    mosaic_dir = Path(f"temp.svalbard/filt/svalbard/mosaics_3584")
    fixes = adsvalbard.vert_coreg.get_temporal_biascorr_fixes()
    for i, (year, bounds) in enumerate(fixes):
        filepaths = {
            "ocean": mosaic_dir / f"dem_{year}_ocean.vrt",
            "stable_terrain": CONSTANTS.temp_dir / "stable_terrain.tif",
            "quality": mosaic_dir / f"dem_{year}_quality.vrt",
            "nmad": mosaic_dir / f"trend_2019-2024_nmad.vrt",
            "slope": mosaic_dir / f"trend_2019-2024_slope.vrt",
        }
        print(f"Fix {i+1} / {len(fixes)}: {rio.coords.BoundingBox(*bounds)}")
        chunks = all_chunks[all_chunks.intersects(shapely.geometry.box(*bounds))]
        bounds = rio.coords.BoundingBox(*chunks.total_bounds)

        with rio.open(filepaths["ocean"]) as raster:
            window = rio.windows.from_bounds(*bounds, raster.transform)
            if verbose:
                print("Reading ocean")
            ocean = raster.read(1, window=window, boundless=True) == 1

            # arr_bounds = rio.windows.bounds(window, raster.transform)
            if verbose:
                print("Constructing coords")
            # x_coords, y_coords = np.meshgrid(
            #     np.linspace(arr_bounds[0] + raster.res[0] / 2, arr_bounds[2] - raster.res[0] / 2, ocean.shape[1]), 
            #     np.linspace(arr_bounds[1] + raster.res[1] / 2, arr_bounds[3] - raster.res[1] / 2, ocean.shape[0])[::-1], 
            # )

            window_transform = rio.windows.transform(window, raster.transform)
        with rio.open(filepaths["stable_terrain"]) as raster:
            if verbose:
                print("Reading stable")
            window = rio.windows.from_bounds(*bounds, raster.transform)
            stable_terrain = raster.read(1, window=window, boundless=True) == 1
        
        glaciers = (~ocean) & (~stable_terrain)
        with rio.open(filepaths["quality"]) as raster:
            window = rio.windows.from_bounds(*bounds, raster.transform)
            missing_data = (raster.read(1, window=window, boundless=True) == 0)

        
        patch = missing_data & glaciers

        gap_distance = scipy.ndimage.distance_transform_edt(patch) * raster.res[0]
        err_d = np.zeros(patch.shape, dtype="float64")
        err_d[patch] = distance_err_model(gap_distance[patch])


        if include_neff_standardization:
            # Label and process each patch part individually
            labeled, _ = scipy.ndimage.label(patch, structure=np.ones((3, 3)))
            areas = pd.Series(*np.unique(labeled, return_counts=True)[::-1]).drop([0])
            areas = (areas *  (raster.res[0] ** 2) / 1e6).sort_values()

            # Merge all patches below 1 km² because it can be tens of thousands of individual pixels
            to_merge = areas[areas < 1].index
            labeled[np.isin(labeled, to_merge)] = areas.index.max() + 1
            areas[areas.index.max() + 1] = areas[to_merge].sum()
            areas.drop(to_merge, inplace=True)

            for idx, gap_area_km2 in areas.items():
                err_area = area_err_model(gap_area_km2)
                neff = neff_model(gap_area_km2)
                target_mean_err = err_area * np.sqrt(neff)
                d_mean_err = np.nanmean(err_d[labeled == idx])
                inflation = target_mean_err / d_mean_err
                err_d[labeled == idx] *= inflation

    
                print(f"{target_mean_err=:.2f}, {d_mean_err=:.2f}, {neff=:.2f}, {gap_area_km2=:.2f}") 


        for _, chunk in chunks.iterrows():

            se_path = Path(f"temp.svalbard/filt/svalbard/chunks_3584/{chunk['chunk_id']}/{chunk['chunk_id']}_trend_2019-2024_slope.tif")

            out_path = Path(f"temp.svalbard/uncertainty/chunks_3584/{se_path.stem.replace('_slope', '_slope_tcorr_err')}.tif")
            with rio.open(se_path) as reference:

                data_window = rio.windows.from_bounds(*reference.bounds, transform=window_transform).round_offsets().round_lengths()

                meta = reference.meta

            print(meta)

            adsvalbard.mosaicking.write_formatted(
                "slope_",  # This will trigger the same write profile as the slope rasters
                out_path,
                err_d[*data_window.toslices()],
                params=meta,
                redo=True,
            )
            # with rio.open(se_path.with_stem(se_path.stem.replace("_slope", "_slope_tcorr_err"))) as raster:

                # print(data_window.toslices())
            

        # plt.subplot(1,2, 1)
        # plt.imshow(gap_distance)
        # plt.subplot(1,2, 2)
        # plt.imshow(err_d)

        # plt.show()

    # dist_bins = np.linspace(-0.01, distances.max() + 0.01, 50)
    # bin_centers = (dist_bins[1:] + dist_bins[:-1]) / 2
    # digitized = np.digitize(distances, bins=dist_bins)

    # import matplotlib.pyplot as plt
    # for idx in np.unique(digitized):
    #     vals = diffs[digitized == idx]

    #     nmad = get_nmad(vals)

    #     plt.scatter(bin_centers[idx - 1], nmad)

    # plt.show()


    # diff = trend_with_2024 - trend_without_2024

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 3, 1)
    # plt.imshow(trend_without_2024, vmin=-1, vmax=1, cmap="RdBu")
    # plt.subplot(1, 3, 2)
    # plt.imshow(trend_with_2024, vmin=-1, vmax=1, cmap="RdBu")
    # plt.subplot(1, 3, 3)
    # plt.imshow(diff, vmin=-0.2, vmax=0.2, cmap="RdBu")
    # plt.show()

            


if __name__ == "__main__":
    main()
