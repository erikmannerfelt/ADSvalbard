import concurrent.futures
import datetime
import os
import tempfile
from typing import Any

import geopandas as gpd
import lxml.etree as ET
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import xarray as xr
from tqdm import tqdm

import adsvalbard.caching
import adsvalbard.inputs
import adsvalbard.utilities


@adsvalbard.caching.cache_json(cache_label="cache_label")
def get_all_geocells_metadata(
    cache_label=datetime.datetime.utcnow().strftime("%Y_%m"),
) -> dict[str, Any]:
    url = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/strips/s2s041/2m.json"

    return adsvalbard.utilities.download_json(url=url)


@adsvalbard.caching.cache_json(subdir="geocell_meta")
def get_geocell_metadata(url: str) -> dict[str, Any]:
    return adsvalbard.utilities.download_json(url=url)


@adsvalbard.caching.cache_json(subdir="strip_meta")
def get_strip_metadata(url: str) -> dict[str, Any]:
    return adsvalbard.utilities.download_json(url=url)


@adsvalbard.caching.cache_feather(cache_label="region_label")
def get_strips(region_label: str = "nordenskiold") -> gpd.GeoDataFrame:
    bounds = adsvalbard.utilities.get_bounds(region=region_label)
    crs = adsvalbard.utilities.get_crs()

    bounds_wgs = (
        gpd.GeoSeries(shapely.geometry.box(*list(bounds)), crs=crs)
        .to_crs(4326)
        .total_bounds
    )

    data_json = get_all_geocells_metadata()

    data = pd.DataFrame.from_records(data_json["links"])
    data = data[data["title"].str.contains("Geocell")]
    data["title"] = data["title"].str.replace("Geocell ", "")

    data["lat"] = data["title"].str.slice(1, 3).astype(int)
    data["lon"] = data["title"].str.slice(4, 7).astype(int) * (
        (data["title"].str.slice(-4, -3) == "e") * 2 - 1
    )
    data = data[
        (data["lon"] >= np.floor(bounds_wgs[0]))
        & (data["lon"] <= np.ceil(bounds_wgs[2]))
        & (data["lat"] >= np.floor(bounds_wgs[1]))
        & (data["lat"] <= np.ceil(bounds_wgs[3]))
    ]

    print("Downloading geocell metadata")
    cell_files = []
    for _, row in data.iterrows():
        cell_meta = get_geocell_metadata(row["href"])
        files = pd.DataFrame.from_records(cell_meta["links"])[["title", "href"]]
        files = files[files["title"].str.contains("SETSM")]
        cell_files.append(files)

    files = pd.concat(cell_files).drop_duplicates("title").sort_values("title")

    def download_strips(url, progress_bar=None):
        meta = get_strip_metadata(url)
        if progress_bar is not None:
            progress_bar.update()

        return meta

    with tqdm(total=len(files), desc="Downloading metadata") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_strip_meta = list(
                executor.map(
                    lambda url: download_strips(url, progress_bar), files["href"].values
                )
            )

    def format_strip(strip_meta: dict[str, Any]) -> dict[str, Any]:
        out = strip_meta["properties"]
        del out["description"]
        for i, coord in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            out[f"bbox_{coord}"] = strip_meta["bbox"][i]

        out["geometry"] = strip_meta["geometry"]
        self_link = [link for link in strip_meta["links"] if link["rel"] == "self"][0]
        out["dem"] = os.path.join(
            os.path.dirname(self_link["href"]), strip_meta["assets"]["dem"]["href"][2:]
        )
        out["matchtag"] = os.path.join(
            os.path.dirname(self_link["href"]),
            strip_meta["assets"]["matchtag"]["href"][2:],
        )

        return out

    strips = pd.DataFrame.from_records(
        [format_strip(meta) for meta in all_strip_meta]
    )
    strips["geometry"] = strips["geometry"].apply(shapely.geometry.shape)
    strips = gpd.GeoDataFrame(strips.drop(columns=["geometry"]), geometry=strips["geometry"], crs=rio.CRS.from_epsg(4326))
    strips["geometry"] = gpd.GeoSeries.from_wkb(
        strips["geometry"].to_wkb(output_dimension=2)
    )
    strips = strips.to_crs(crs)

    for col in ["pgc:image_ids", "pgc:avg_sun_elevs", "instruments"]:
        strips[col] = strips[col].apply(lambda col_list: ",".join(map(str, col_list)))

    return strips


