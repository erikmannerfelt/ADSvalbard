import geopandas as gpd
import numpy as np
import shapely
import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
import rasterio as rio
import adsvalbard.utilities
import adsvalbard.inputs
from typing import Any
from pathlib import Path

@adsvalbard.utilities.cache_json
def get_all_geocells_metadata() -> dict[str, Any]:
    url = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/strips/s2s041/2m.json"

    return adsvalbard.utilities.download_json(url=url)


@adsvalbard.utilities.cache_json_subdir("geocell_meta")
def get_geocell_metadata(url: str) -> dict[str, Any]:
    return adsvalbard.utilities.download_json(url=url)


@adsvalbard.utilities.cache_json_subdir("strip_meta")
def get_strip_metadata(url: str) -> dict[str, Any]:
    return adsvalbard.utilities.download_json(url=url)
    
@adsvalbard.utilities.cache_feather
def get_strips(cache_label: str = "nordenskiold") -> gpd.GeoDataFrame:
    region = cache_label
    bounds = adsvalbard.utilities.get_bounds(region=region)
    crs = adsvalbard.utilities.get_crs()

    bounds_wgs = gpd.GeoSeries(shapely.geometry.box(*list(bounds)), crs=crs).to_crs(4326).total_bounds

    data_json = get_all_geocells_metadata()

    data = pd.DataFrame.from_records(data_json["links"])
    data = data[data["title"].str.contains("Geocell")]
    data["title"] = data["title"].str.replace("Geocell ", "")

    data["lat"] = data["title"].str.slice(1, 3).astype(int)
    data["lon"] = data["title"].str.slice(4, 7).astype(int) * ((data["title"].str.slice(-4, -3) == "e") * 2 - 1)
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
            all_strip_meta = list(executor.map(lambda url: download_strips(url, progress_bar), files["href"].values))


    def format_strip(strip_meta: dict[str, Any]) -> dict[str, Any]:
        out = strip_meta["properties"]
        del out["description"]
        for i, coord in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            out[f"bbox_{coord}"] = strip_meta["bbox"][i]

        out["geometry"] = strip_meta["geometry"]
        self_link = [link for link in strip_meta["links"] if link["rel"] == "self"][0]
        out["dem"] = os.path.join(os.path.dirname(self_link["href"]), strip_meta["assets"]["dem"]["href"][2:])
        out["matchtag"] = os.path.join(os.path.dirname(self_link["href"]), strip_meta["assets"]["matchtag"]["href"][2:])

        return out

    strips = gpd.GeoDataFrame.from_records([format_strip(meta) for meta in all_strip_meta])
    strips["geometry"] = strips["geometry"].apply(shapely.geometry.shape)
    strips.crs = rio.CRS.from_epsg(4326)
    strips["geometry"] = gpd.GeoSeries.from_wkb(strips["geometry"].to_wkb(output_dimension=2))
    strips = strips.to_crs(crs)

    for col in ["pgc:image_ids", "pgc:avg_sun_elevs", "instruments"]:
        strips[col] = strips[col].apply(lambda col_list: ",".join(map(str, col_list)))

    return strips


def download_arcticdem(dem_data: pd.Series) -> tuple[Path, Path]:
    data_dir = adsvalbard.utilities.get_data_dir("ArcticDEM")

    filepaths = []
    for kind in ["dem", "matchtag"]:
        filepath = adsvalbard.utilities.download_large_file(dem_data[kind], directory=data_dir)
        filepaths.append(filepath)

    return filepaths[0], filepaths[1]