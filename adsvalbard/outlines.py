import datetime
import time

import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import scipy
import shapely
import tqdm
import xarray as xr

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS


def interpolate_masks(
    outline_0: shapely.geometry.Polygon,
    outline_1: shapely.geometry.Polygon,
    time_0: datetime.datetime,
    time_1: datetime.datetime,
    times_interp: list[datetime.datetime],
    res: float = CONSTANTS.res,
    bounds: rasterio.coords.BoundingBox | None = None,
    verbose: bool = True,
) -> list[np.ndarray]:

    start_time = time.time()
    def print_progress(message):
        if not verbose:
            return
        print(f"+{time.time() - start_time:.1f} s: {message}")
        
    print_progress("Starting new interp")
    time0_float = np.array([time_0]).astype("datetime64[ns]").astype(float)[0]
    time1_float = np.array([time_1]).astype("datetime64[ns]").astype(float)[0]
    times_interp_float = np.array(times_interp).astype("datetime64[ns]").astype(float)
    times_frac = (times_interp_float - time0_float) / (time1_float - time0_float)

    diff = shapely.difference(outline_0, outline_1)
    diff2 = shapely.difference(outline_1, outline_0)

    debug_plot = False
    if debug_plot:
        fig = plt.figure(figsize=(8, 6))
        grid_shape = (6, 6)
        axis = plt.subplot2grid(grid_shape, (0, 0), colspan=2, rowspan=3)
        axis.plot(*outline_0.exterior.xy)
        plt.text(0.5, 1., f"{time_0.date()} to {time_1.date()}", va="bottom", ha="center", transform=axis.transAxes)

        ylim = min(diff2.bounds[1], diff.bounds[1]), max(diff2.bounds[3], diff.bounds[3])
        xlim = min(diff2.bounds[0], diff.bounds[0]), max(diff2.bounds[2], diff.bounds[2])

    if "Multi" in diff2.geom_type:
        diff2 = list(diff2.geoms)
    else:
        diff2 = [diff2]

    diff2 = [outline for outline in diff2 if not shapely.overlaps(diff, outline)]

    if "Multi" in diff.geom_type:
        diff = list(diff.geoms)
    else:
        diff = [diff]

    outline_points = []
    for outline in [*diff, *diff2]:
        # plt.gca().add_patch(plt.Polygon(outline.exterior.xy))
        # plt.plot(*outline.exterior.xy)
        if debug_plot:
            gpd.GeoSeries(outline).plot(ax=axis, color="red")
        for dist in np.arange(0, outline.exterior.length, step=res):

            point = outline.exterior.interpolate(dist)

            if outline_0.exterior.distance(point) < 1e-8:
                outline_points.append(
                    {
                        "value": 0,
                        "geometry": point,
                    }
                )
            else:
                outline_points.append(
                    {
                        "value": 1,
                        "geometry": point,
                    }
                )
                
    if debug_plot:
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_axis_off()
    # plt.close()
    # raise NotImplementedError()
    # for key, outline, date in [("pre", outline_0, time_0), ("post", outline_1, time_1)]:
    #     for dist in np.arange(0, outline.exterior.length, step=res):

    #         point = outline.exterior.interpolate(dist)

    #         # if shapely.touches(point, outline_0) and shapely.touches(point, outline_1):
    #         #     continue
    #         outline_points.append(
    #             {
    #                 "key": key,
    #                 "date": date,
    #                 "geometry": point,
    #             }
    #         )

    outline_points = gpd.GeoDataFrame(outline_points)


    # outline_points["value"] = (outline_points["key"] == "post").astype(int)
    outline_points["x"] = outline_points["geometry"].x
    outline_points["y"] = outline_points["geometry"].y

    # plt.scatter(outline_points["x"], outline_points["y"], c=outline_points["value"])
    # plt.colorbar()
    # plt.plot(*diff[0].exterior.xy)
    # plt.show()

    if bounds is None:
        dst_bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*outline_points.total_bounds), buffer=CONSTANTS.res * 2)
    else:
        dst_bounds = bounds
    
    shape = adsvalbard.utilities.get_shape(dst_bounds, [CONSTANTS.res] * 2)

    # print(shape)

    # raise NotImplementedError()

    # eastings, northings = np.meshgrid(
    #     np.linspace(dst_bounds.left +  res/ 2, dst_bounds.right - res / 2, shape[1]),
    #     np.linspace(dst_bounds.bottom + res/ 2, dst_bounds.top - res / 2, shape[0])[::-1],
    # )
    eastings, northings = adsvalbard.utilities.generate_eastings_northings(bounds=dst_bounds, shape=shape)

    transform = adsvalbard.utilities.get_transform(bounds=dst_bounds,res=[res] * 2)

    mask_pre = rasterio.features.rasterize([outline_0], out_shape=shape, transform=transform) == 1
    mask_post = rasterio.features.rasterize([outline_1], out_shape=shape, transform=transform) == 1

    smallest = mask_pre & mask_post
    largest = mask_pre | mask_post

    # plt.scatter(outline_points["x"], outline_points["y"], c=outline_points["value"])

    change_region = largest & ~smallest

    # plt.imshow(change_region)
    # plt.show()
    

    # plt.show()

    print_progress("Creating RBF")
    interp = scipy.interpolate.RBFInterpolator(outline_points[["x", "y"]], outline_points["value"], kernel="linear")

    print_progress("Interpolating")

    mask_float = np.zeros(eastings.shape, dtype=float)

    mask_float[change_region] = interp(np.transpose([eastings[change_region], northings[change_region]]))
    # mask_float = interp(np.transpose([eastings.ravel(), northings.ravel()])).reshape(eastings.shape)
    print_progress("Done. Applying interpolation")
    mask_float[smallest] = 0.
    mask_float[~largest] = 1.

    if debug_plot:
        axis = plt.subplot2grid(grid_shape, (3, 0), colspan=2, rowspan=3)
        extent = [dst_bounds.left, dst_bounds.right, dst_bounds.bottom, dst_bounds.top]
        axis.imshow(np.ma.masked_array(mask_float, mask=~largest | smallest), extent=extent)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_axis_off()
        plt.text(0.5, 0.01, f"Interpolated difference", va="bottom", ha="center", transform=axis.transAxes)

    masks = []
    i = -1
    for frac in times_frac:
        new_mask = np.bitwise_xor(mask_pre, (mask_float < frac) & ~smallest)

        masks.append(new_mask)

        if i >= 0 and debug_plot:
            row = int(i / 4)
            axis = plt.subplot2grid(grid_shape, (row * 2, 2 + i % 4), rowspan=2) 
            axis.imshow(new_mask, extent=extent, cmap="Greys")
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_axis_off()
            plt.text(0.5, 0.99, f"{frac:.2f} ({str(times_interp[i])[:10]})", va="bottom", ha="center", transform=axis.transAxes, fontsize=8)
        # plt.show()
        i += 1

    if debug_plot:
        plt.subplots_adjust(left=0, bottom=0.03, right=0.971, top=0.96)
        plt.savefig("figures/interpolated_masks.jpg", dpi=600)
        plt.show()

    return masks


def generate_interpolated_times(start_year: int, end_year: int, freq: str = "MS"):
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq=freq).to_numpy()

    # TODO: Add bfill and ffill
    # times = times[(times > outlines["src_date"].min()) & (times < outlines["src_date"].max())]

    return times

    # outlines = all_outlines[all_outlines["glac_name"] == "Scheelebreen"]

    # print(all_outlines.groupby("glac_name").first().geometry.area.sort_values() / 1e6)

    
    

def generate_masks(outlines: gpd.GeoDataFrame, start_year = 2012, end_year = 2023):

    outlines = outlines.sort_values("src_date")

    bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*outlines.total_bounds), buffer=CONSTANTS.res * 2)

    # times = pd.arrays.DatetimeArray._from_sequence(pd.DatetimeIndex([f"{start_year}-01-01", f"{end_year}-12-31"], freq="ME"))
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS").to_numpy()

    # TODO: Add bfill and ffill
    times = times[(times > outlines["src_date"].min()) & (times < outlines["src_date"].max())]


    masks = {}
    for i in tqdm.tqdm(range(1, outlines.shape[0]), total=outlines.shape[0] - 1):
        # if i <= 2:
        #     print("DEBUG: REMOVE THIS")
        #     continue
        outline_0 = outlines.iloc[i - 1]
        outline_1 = outlines.iloc[i]
        times_interp = times[(times > outline_0["src_date"]) & (times < outline_1["src_date"])]

        new_masks = interpolate_masks(
            outline_0=outline_0["geometry"],
            outline_1=outline_1["geometry"],
            time_0=outline_0["src_date"],
            time_1=outline_1["src_date"],
            times_interp=times_interp,
            res = CONSTANTS.res,
            bounds=bounds,
            verbose=False,
        )

        for j, mask in enumerate(new_masks):

            if times_interp[j] in masks:
                continue
            masks[times_interp[j]] = mask
            # plt.title(times_interp[j])
            # plt.imshow(mask)
            # plt.show()

   
    shape = adsvalbard.utilities.get_shape(bounds, [CONSTANTS.res] * 2)
    eastings, northings = adsvalbard.utilities.generate_eastings_northings(bounds=bounds, shape=shape)

    outlines = xr.DataArray(
        np.array(list(masks.values())),
        coords=[("time", list(masks.keys())), ("y", northings[:, 0]), ("x", eastings[0, :])],
    )

    return outlines
   

    return

    for i, (time, mask) in enumerate(masks.items()):

        aspect = mask.shape[0] / mask.shape[1]

        figsize = 5 / aspect, 5 

        plt.figure(figsize=figsize)
        plt.title(str(time)[:10])
        plt.imshow(mask, cmap="Greys")
        plt.gca().set_axis_off()
        plt.savefig(f"tmp.fig/scheele_{str(i).zfill(5)}.jpg", dpi=300)
        plt.close()
       
    return

    # test_time = pd.Timestamp("2017-06-01")
    # time_diff = outlines["src_date"] - test_time

    # outline_pre = outlines.loc[time_diff[time_diff.dt.total_seconds() < 0].sort_values().index[-1]]
    # outline_post = outlines.loc[time_diff[time_diff.dt.total_seconds() > 0].sort_values().index[0]]
    outline_pre = outlines.iloc[0]
    outline_post = outlines.iloc[-1]


    times_interp = np.linspace(np.array([outline_pre["src_date"].to_numpy()]).astype(float)[0], np.array([outline_post["src_date"].to_numpy()]).astype(float)[0], 10).astype("datetime64[ms]")

    interpolate_masks(
        outline_0=outline_pre["geometry"],
        outline_1=outline_post["geometry"],
        time_0=outline_pre["src_date"],
        time_1=outline_post["src_date"],
        times_interp=times_interp,
        res = CONSTANTS.res,
        bounds=bounds,
    )

    return

    outline_points = []
    for key, outline in [("pre", outline_pre), ("post", outline_post)]:
        for dist in np.arange(0, outline["geometry"].exterior.length, step=CONSTANTS.res):

            outline_points.append(
                {
                    "key": key,
                    "date": outline["src_date"],
                    "geometry": outline["geometry"].exterior.interpolate(dist),
                }
            )

    outline_points = gpd.GeoDataFrame(outline_points, crs=CONSTANTS.crs_epsg)

    outline_points["value"] = (outline_points["key"] == "post").astype(int)
    outline_points["x"] = outline_points["geometry"].x
    outline_points["y"] = outline_points["geometry"].y


    bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*outline_points.total_bounds), buffer=CONSTANTS.res * 2)
    
    shape = adsvalbard.utilities.get_shape(bounds, [CONSTANTS.res] * 2)

    eastings, northings = np.meshgrid(
        np.linspace(bounds.left +  CONSTANTS.res/ 2, bounds.right - CONSTANTS.res / 2, shape[1]),
        np.linspace(bounds.bottom +  CONSTANTS.res/ 2, bounds.top - CONSTANTS.res / 2, shape[0])[::-1],
    )

    transform = adsvalbard.utilities.get_transform(bounds=bounds,res=[CONSTANTS.res] * 2)

    mask_pre = rasterio.features.rasterize([outline_pre["geometry"]], out_shape=shape, transform=transform) == 1
    mask_post = rasterio.features.rasterize([outline_post["geometry"]], out_shape=shape, transform=transform) == 1

    smallest = mask_pre & mask_post
    largest = mask_pre | mask_post

    # plt.scatter(outline_points["x"], outline_points["y"], c=outline_points["value"])

    # plt.show()

    interp = scipy.interpolate.RBFInterpolator(outline_points[["x", "y"]], outline_points["value"], kernel="linear")

    mask_float = interp(np.transpose([eastings.ravel(), northings.ravel()])).reshape(eastings.shape)
    mask_float[smallest] = 0.
    mask_float[~largest] = 1.


    threshold = 0.5

    # new_mask = largest & ~(mask_float < threshold)


    for n in np.linspace(0, 1, 9):
        new_mask = np.bitwise_xor(mask_pre, (mask_float < n) & ~smallest)
        plt.title(n)
        plt.imshow(new_mask)
        plt.savefig(f"tmp.fig/mask_{str(int(n * 100)).zfill(2)}.jpg", dpi=300)

        plt.close()

    plt.subplot(1, 3, 1)
    plt.imshow(mask_pre)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_post)
    plt.subplot(1, 3, 3)
    plt.imshow(np.ma.masked_array(mask_float, mask=(~largest) | smallest), vmin=0, vmax=1)
    plt.show()
    # return
    # mask = scipy.interpolate.griddata(
    #     points=outline_points[["x", "y"]],
    #     values=outline_points["value"],
    #     xi=(eastings, northings),

    # )


    

    

    # print(np.argmax(time_diff))
    # outline_pre = outlines[
    # scipy.interpolate.RBFInterpolator(kernel="linear")
    

def generate_full_glacier_mask(region: str = "heerland", start_year = 2021, end_year=2023):

    all_outlines = gpd.read_file("shapes/glacier_outlines.sqlite")
    all_outlines["area_km2"] = all_outlines["geometry"].area / 1e6
    all_outlines.sort_values("area_km2", inplace=True)

    
    bounds = CONSTANTS.regions[region]
    shape = adsvalbard.utilities.get_shape(rasterio.coords.BoundingBox(**bounds), [CONSTANTS.res] * 2)
    # eastings, northings = adsvalbard.utilities.generate_eastings_northings(bounds=bounds, shape=shape)
    xr_coords = {
        "time": generate_interpolated_times(start_year=start_year, end_year=end_year),
        "y": np.linspace(bounds["bottom"] + CONSTANTS.res / 2, bounds["top"] - CONSTANTS.res / 2, shape[0])[::-1],
        "x": np.linspace(bounds["left"] + CONSTANTS.res / 2, bounds["right"] - CONSTANTS.res / 2, shape[1]),
    }

    out_mask = da.zeros((xr_coords["time"].shape[0],) + shape, dtype="uint8")
    # out_mask = xr.DataArray(
    #     da.zeros((xr_coords["time"].shape[0],) + shape, dtype="uint8"),
    #     coords=xr_coords,
    # ).to_dataset(name="index_mask")

    # out_mask["glacier_index"] = np.sort(all_outlines["mask_id"].unique()).astype("uint8")

    extra = []
    for i, (glac_name, outlines) in enumerate(all_outlines.groupby("glac_name")):

        if outlines.iloc[0]["area_km2"] > 2:
            continue
        if len(extra) >= 4:
            break

        masks = generate_masks(outlines=outlines, start_year=start_year, end_year=end_year).reindex(time=xr_coords["time"]).bfill("time").ffill("time")

        attrs = {key: outlines.iloc[0][key] for key in ["rgi_id", "glims_id", "glac_name", "mask_id"]} | {"n_outlines": outlines.shape[0], "first_outline": outlines["src_date"].min(), "last_outline": outlines["src_date"].max()}

        new_extra = xr.Dataset().assign_coords(bounds=["xmin", "ymin", "xmax", "ymax"], glacier_index=[attrs["mask_id"]])
        new_extra["bounding_box"] = ("glacier_index", "bounds"), [[masks["x"].min().item(), masks["y"].min().item(), masks["x"].max().item(), masks["y"].max().item()]]

        for key in attrs:
            if "mask_id" == key:
                continue
            new_extra[key] = ("glacier_index",), [attrs[key]]

        extra.append(new_extra)

        # out_mask.loc[{"x": masks.x, "y": masks.y}] += masks.astype("uint8") * np.uint8(attrs["mask_id"])

        xslice = np.argwhere((xr_coords["x"] >= masks.x.min().item()) & ((xr_coords["x"] <= masks.x.max().item()))).ravel()[[0, -1]]
        yslice = np.argwhere((xr_coords["y"] >= masks.y.min().item()) & ((xr_coords["y"] <= masks.y.max().item()))).ravel()[[0, -1]]

        out_mask[:, yslice[0]:yslice[1] + 1, xslice[0]:xslice[1] + 1] += masks.values.astype("uint8") * np.uint8(attrs["mask_id"])
        # print(pd.Series(xr_coords["x"]).isin(masks.x).map(lambda x: x[x], axis=1))
        # out_mask["index_mask"].loc[{"x": masks.x, "y": masks.y}]+= masks.astype("uint8") * np.uint8(attrs["mask_id"])

        
    out = xr.concat(extra, dim="glacier_index").assign_coords(**xr_coords)
    out["index_mask"] = ("time", "y", "x"), out_mask

    print(out)
