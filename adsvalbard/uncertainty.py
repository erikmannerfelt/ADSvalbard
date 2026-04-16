from adsvalbard import outlines, vert_coreg
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
    Path(f"temp.svalbard/bad_dems{prefix}.txt").write_text("\n".join(bad_titles))

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
    

def main(redo: bool = False):
    from osgeo import gdal
    gdal.UseExceptions()

    temp_dir = get_temp_dir()
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
        out_paths = []

        for chunk_dir in tqdm.tqdm(chunk_dirs, desc=f"Applying err functions for {pred_col}"):
            try:
                out_path = apply_function(chunk_dir, func=func, pred_col=pred_col, var_cols=var_cols, redo=redo)
                out_paths.append(str(out_path))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error on {chunk_dir}: {e}")

        gdal.BuildVRT(
            str(temp_dir / f"{pred_col}_err_sigma1.vrt"),
            out_paths,
        )

    return
    points["dh_err_pred"] = np.hypot(func(points[var_cols]), points[se_col]) 

    vmin, vmax = np.nanpercentile(points["dh_err_pred"], [2, 98])
    plt.scatter(points.geometry.x, points.geometry.y, c=points["dh_err_pred"], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
