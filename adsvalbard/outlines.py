import geopandas as gpd
import scipy
import pandas as pd
import numpy as np
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
import datetime
import tqdm

from adsvalbard.constants import CONSTANTS
import adsvalbard.utilities
import shapely


def interpolate_masks(
    outline_0: shapely.geometry.Polygon,
    outline_1: shapely.geometry.Polygon,
    time_0: datetime.datetime,
    time_1: datetime.datetime,
    times_interp: list[datetime.datetime],
    res: float = CONSTANTS.res,
    bounds: rasterio.coords.BoundingBox | None = None
) -> list[np.ndarray]:
    print("Starting new interp")
    time0_float = np.array([time_0]).astype("datetime64[ns]").astype(float)[0]
    time1_float = np.array([time_1]).astype("datetime64[ns]").astype(float)[0]
    times_interp_float = np.array(times_interp).astype("datetime64[ns]").astype(float)
    times_frac = (times_interp_float - time0_float) / (time1_float - time0_float)


    diff = shapely.difference(outline_1, outline_0)


    if "Multi" in diff.geom_type:
        diff = list(diff.geoms)
    else:
        diff = [diff]

    outline_points = []
    for outline in diff:
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

    eastings, northings = np.meshgrid(
        np.linspace(dst_bounds.left +  res/ 2, dst_bounds.right - res / 2, shape[1]),
        np.linspace(dst_bounds.bottom + res/ 2, dst_bounds.top - res / 2, shape[0])[::-1],
    )

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

    print("Creating RBF")
    interp = scipy.interpolate.RBFInterpolator(outline_points[["x", "y"]], outline_points["value"], kernel="linear")

    print("Interpolating")

    mask_float = np.zeros(eastings.shape, dtype=float)

    mask_float[change_region] = interp(np.transpose([eastings[change_region], northings[change_region]]))
    # mask_float = interp(np.transpose([eastings.ravel(), northings.ravel()])).reshape(eastings.shape)
    print("Done")
    mask_float[smallest] = 0.
    mask_float[~largest] = 1.

    masks = []
    for frac in times_frac:
        new_mask = np.bitwise_xor(mask_pre, (mask_float < frac) & ~smallest)

        masks.append(new_mask)
        # plt.imshow(new_mask)
        # plt.show()

    return masks


def generate_masks(start_year = 2012, end_year = 2023):

    outlines = gpd.read_file("shapes/glacier_outlines.sqlite")

    outlines = outlines[outlines["glac_name"] == "Vallåkrabreen"].sort_values("src_date")

    bounds = adsvalbard.utilities.align_bounds(rasterio.coords.BoundingBox(*outlines.total_bounds), buffer=CONSTANTS.res * 2)

    # times = pd.arrays.DatetimeArray._from_sequence(pd.DatetimeIndex([f"{start_year}-01-01", f"{end_year}-12-31"], freq="ME"))
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS").to_numpy()

    # TODO: Add bfill and ffill
    times = times[(times > outlines["src_date"].min()) & (times < outlines["src_date"].max())]


    masks = {}
    for i in tqdm.tqdm(range(1, outlines.shape[0]), total=outlines.shape[0] - 1):
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
        )

        for j, mask in enumerate(new_masks):

            if times_interp[j] in masks:
                continue
            masks[times_interp[j]] = mask
            # plt.title(times_interp[j])
            # plt.imshow(mask)
            # plt.show()


    for i, (time, mask) in enumerate(masks.items()):

        aspect = mask.shape[0] / mask.shape[1]

        figsize = 5 / aspect, 5 

        plt.figure(figsize=figsize)
        plt.title(str(time)[:11])
        plt.imshow(mask)
        plt.savefig(f"tmp.fig/vallakra_{str(i).zfill(5)}.jpg", dpi=300)
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
    




