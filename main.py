from pathlib import Path
from pyproj import CRS
import geopandas as gpd
import rasterio as rio
import rasterio.features
import rasterio.warp
import os
import xdem
import geoutils as gu
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import re
import json
from tqdm import tqdm
from osgeo import gdal
import datetime
import pandas as pd
import shapely.geometry
import requests
import concurrent.futures


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON encoder that properly serializes numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_data_urls():
    np_dem_base_url = "https://public.data.npolar.no/kartdata/S0_Terrengmodell/Delmodell/"

    data_dir = get_data_dir()

    urls = {
        "strip_metadata": [
            (
                "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/strips/s2s041/2m.json",
                "geocell_metadata.json",
            )
        ],
        "NP_DEMs": [
            np_dem_base_url + url
            for url in [
                "NP_S0_DTM5_2008_13652_33.zip",
                "NP_S0_DTM5_2009_13824_33.zip",
                "NP_S0_DTM5_2009_13835_33.zip",
                "NP_S0_DTM5_2010_13923_33.zip" "NP_S0_DTM5_2010_13836_33.zip",
                "NP_S0_DTM5_2011_25162_33.zip",
                "NP_S0_DTM5_2012_25235_33.zip",
            ]
        ],
        "outlines": [
            "https://public.data.npolar.no/kartdata/NP_S100_SHP.zip",
            (
                "https://api.npolar.no/dataset/"
                + "f6afca5c-6c95-4345-9e52-cfe2f24c7078/_file/3df9512e5a73841b1a23c38cf4e815e3",
                "GAO_SfM_1936_1938.zip",
            ),
        ],
    }

    for key in urls:
        for i, entry in enumerate(urls[key]):
            if isinstance(entry, tuple):
                filename = entry[1]
                url = entry[0]
            else:
                filename = os.path.basename(entry)
                url = entry

            urls[key][i] = (url, data_dir.joinpath(key).joinpath(filename))

    return urls


def get_strips() -> gpd.GeoDataFrame:

    temp_dir = get_temp_dir()

    output_path = temp_dir.joinpath("strip-meta.feather")

    if output_path.is_file():
        return gpd.read_feather(output_path)

    bounds = get_bounds()
    crs = get_crs()
    bounds_wgs = gpd.GeoSeries(shapely.geometry.box(*list(bounds)), crs=get_crs()).to_crs(4326).total_bounds

    filepath = [v[1] for v in get_data_urls()["strip_metadata"] if "geocell_metadata" in str(v[1])][0]

    with open(filepath) as infile:
        data_json = json.load(infile)

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

    cell_files = []
    for _, row in data.iterrows():

        cell_filepath = filepath.parent.joinpath(os.path.basename(row["href"]))
        if not cell_filepath.is_file():
            response = requests.get(row["href"])

            if response.status_code != 200:
                raise ValueError(f"{response.status_code}: {response.content}")

            with open(cell_filepath, "wb") as outfile:
                outfile.write(response.content)

        with open(cell_filepath) as infile:
            files = pd.DataFrame.from_records(json.load(infile)["links"])[["title", "href"]]
            files = files[files["title"].str.contains("SETSM")]
            cell_files.append(files)


    files = pd.concat(cell_files).drop_duplicates("title").sort_values("title")

    strip_metadata_dir = filepath.parent.joinpath("per-strip")
    strip_metadata_dir.mkdir(exist_ok=True)

    infos = []
    filepaths = set()
    for _, row in files.iterrows():
        strip_filepath = strip_metadata_dir.joinpath(os.path.basename(row["href"]))
        filepaths.add(strip_filepath)
        if strip_filepath.is_file():
            continue
        infos.append((strip_filepath, row["href"]))

    filepaths = list(filepaths)
    def download(info, progress_bar = None):
        strip_filepath, url = info

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: {response.content}")

        with open(strip_filepath, "wb") as outfile:
            outfile.write(response.content)

        if progress_bar is not None:
            progress_bar.update()
        

    if len(infos) > 0:
        
        #for info in tqdm(infos, desc="Downloading metadata"):
        #    download(info)
            
        with tqdm(total=len(infos), desc="Downloading metadata") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
               list(executor.map(lambda i: download(i, progress_bar), infos))


    def load(strip_filepath):

        with open(strip_filepath) as infile:
            data = json.load(infile)

        out = data["properties"]
        del out["description"]
        for i, coord in enumerate(["xmin", "ymin", "xmax", "ymax"]):
            out[f"bbox_{coord}"] = data["bbox"][i]

        out["geometry"] = data["geometry"]
        self_link = [l for l in data["links"] if l["rel"] == "self"][0]
        out["dem"] = os.path.join(os.path.dirname(self_link["href"]), data["assets"]["dem"]["href"][2:])
        out["matchtag"] = os.path.join(os.path.dirname(self_link["href"]), data["assets"]["matchtag"]["href"][2:])


        return out

       
    strips = gpd.GeoDataFrame.from_records([load(fp) for fp in filepaths])
    strips["geometry"] = strips["geometry"].apply(shapely.geometry.shape)
    strips.crs = CRS.from_epsg(4326)
    strips["geometry"] = gpd.GeoSeries.from_wkb(strips["geometry"].to_wkb(output_dimension=2)) 
    strips = strips.to_crs(crs)

    for col in ["pgc:image_ids", "pgc:avg_sun_elevs", "instruments"]:
        strips[col] = strips[col].apply(lambda col_list: ",".join(map(str, col_list)))

    strips.to_feather(output_path)
    return strips


def get_xdem_dem_co() -> dict[str, str]:
    """Get DEM creation options for xdem."""
    return {"COMPRESS": "DEFLATE", "ZLEVEL": "12", "TILED": "YES", "NUM_THREADS": "ALL_CPUS"}


def get_dem_co() -> list[str]:
    """Get DEM creation options for GDAL."""
    return [f"{k}={v}" for k, v in get_xdem_dem_co().items()]


def get_res() -> tuple[float, float]:
    """Get the horizontal/vertical resolution of the output DEMs."""
    return (5.0, 5.0)


def get_bounds(
    region: str = "nordenskiold", res: tuple[float, float] | None = None, half_mod: bool = True
) -> rio.coords.BoundingBox:
    """
    Get the bounding coordinates of the output DEMs.

    Arguments
    ---------
    - region: The selected region (with hardcoded bounds)
    - res: Optional. Override the x/y resolution.
    - half_mod: Whether the modulus should be calculated on half the pixel size (e.g. 50000 at 5m -> 50002.5)

    Returns
    -------
    A bounding box of the region.
    """

    region_bounds = {
        "svalbard": {"left": 341002.5, "bottom": 8455002.5, "right": 905002.5, "top": 8982002.5},
        "nordenskiold": {"left": 443002.5, "bottom": 8626007.5, "right": 560242.5, "top": 8703007.5},
    }

    if res is None:
        res = get_res()

    bounds = region_bounds[region]

    # Ensure that the moduli of the bounds are zero
    for i, bound0 in enumerate([["left", "right"], ["bottom", "top"]]):
        for j, bound in enumerate(bound0):

            mod = (bounds[bound] - (res[i] / 2 if half_mod else 0)) % res[i]

            bounds[bound] = bounds[bound] - mod + (res[i] if i > 0 and mod != 0 else 0)

    return rio.coords.BoundingBox(**bounds)


def get_transform(res: tuple[float, float] | None = None) -> rio.Affine:
    """
    Get the affine transform of the output DEMs.

    Arguments
    ---------
    - res: Optional. Override the x/y resolution.

    Returns
    -------
    An affine transform corresponding to the bounds and the resolution.
    """
    if res is None:
        res = get_res()

    bounds = get_bounds(res=res)
    return rio.transform.from_origin(bounds.left, bounds.top, *res)


def get_shape(res: tuple[float, float] | None = None) -> tuple[int, int]:
    """
    Get the pixel shape (height, width) of the output DEMs

    Arguments
    ---------
    - res: Optional. Override the x/y resolution.

    Returns
    -------
    A tuple (height, width) of the output shape.
    """
    if res is None:
        res = get_res()

    bounds = get_bounds(res=res)

    return (int((bounds.top - bounds.bottom) / res[1]), int((bounds.right - bounds.left) / res[0]))


def get_crs() -> CRS:
    """Get the target Coordinate Reference System (CRS)."""
    return CRS.from_epsg(32633)


def get_temp_dir() -> Path:
    """Get (and create) the path to the directory to store temporary files in."""

    temp_dir = get_data_dir().parent.joinpath("temp")
    temp_dir.mkdir(exist_ok=True, parents=True)
    return temp_dir


def get_data_dir() -> Path:
    """Get the path to the data directory."""
    return Path("data/").absolute()


def build_stable_terrain_mask(verbose: bool = False) -> Path:
    """
    Build a stable terrain mask.

    The mask excludes areas within:

    - 1936 glacier outlines from Geyman et al., 2022
    - Water (rivers/lakes) according to the NPI S100 map
    - Moraines according to the NPI S100 map

    ... and includes all other areas on land according to the NPI S100 map

    Arguments
    ---------
    - verbose: Whether to print updates to the console

    Returns
    -------
    A path to the stable terrain mask.
    """

    data_dir = get_data_dir()
    temp_dir = get_temp_dir()
    out_filepath = temp_dir.joinpath("stable_terrain.tif")

    if out_filepath.is_file():
        return out_filepath

    res = get_res()
    shape = get_shape(res=res)
    transform = get_transform(res=res)
    crs = get_crs()

    geyman_1936 = (
        gpd.read_file(f"zip://{data_dir}/GAO_SfM_1936_1938.zip!GAO_SfM_1936_1938_v3.shp")
        .buffer(0)
        .to_frame()
        .dissolve()
        .to_crs(crs)
    )
    water = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Vann_f.shp").dissolve().to_crs(crs)
    moraines = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Morener_f.shp").dissolve().to_crs(crs)
    land = gpd.read_file(f"zip://{data_dir}/NP_S100_SHP.zip!NP_S100_SHP/S100_Land_f.shp").dissolve().to_crs(crs)

    stable = land.difference(geyman_1936.union(water).union(moraines))
    if verbose:
        print(f"{now_time()}: Loaded stable terrain features in memory")

    rasterized = rasterio.features.rasterize(stable.geometry, out_shape=shape, transform=transform, default_value=1)
    if verbose:
        print(f"{now_time()}: Rasterized features.")

    os.makedirs(out_filepath.parent, exist_ok=True)
    with rio.open(
        out_filepath,
        "w",
        crs=crs,
        transform=transform,
        compress="lzw",
        tiled=True,
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
    ) as raster:
        raster.write(rasterized, 1)
    if verbose:
        print(f"{now_time()}: Saved {out_filepath.name}.")

    return out_filepath


def create_warped_vrt(
    filepath: Path | str, vrt_filepath: Path | str, out_crs: str | CRS, in_crs: str | CRS | None = None
) -> None:
    """
    Create a warped VRT from a raster with a different CRS.

    :param filepath: The path to the raster to create a VRT from.
    :param vrt_filepath: The output path of the VRT.
    :param out_crs: The target CRS of the VRT either as a WKT or a CRS object.
    :param in_crs: The original CRS of the VRT either as a WKT or a CRS object.
    """
    ds = gdal.Open(str(filepath))

    if isinstance(out_crs, CRS):
        out_crs = out_crs.to_wkt()
    if isinstance(in_crs, CRS):
        in_crs = in_crs.to_wkt()

    vrt = gdal.AutoCreateWarpedVRT(ds, in_crs, out_crs, rasterio.warp.Resampling.bilinear)
    vrt.GetDriver().CreateCopy(str(vrt_filepath), vrt)

    del ds
    del vrt


def build_npi_mosaic(verbose: bool = False) -> Path:
    """
    Build a mosaic of tiles downloaded from the NPI.

    Arguments
    ---------
    - verbose: Whether to print updates to the console

    Returns
    -------
    A path to the NPI mosaic.
    """
    data_dir = get_data_dir()
    np_dem_dir = data_dir.joinpath("NP_DEMs")
    temp_dir = get_temp_dir()

    output_path = temp_dir.joinpath("npi_mosaic_clip.tif")

    if output_path.is_file():
        return output_path

    from osgeo import gdal

    crs = get_crs()
    res = get_res()
    bounds = get_bounds(res=res)

    # Generate links to the DEM tiles within their zipfiles
    uris = []
    for filepath in np_dem_dir.glob("NP_S0*.zip"):
        uri = f"/vsizip/{filepath}/{filepath.stem}/{filepath.stem.replace('NP_', '')}.tif"
        uris.append(uri)

    vrt_dir = temp_dir.joinpath("npi_vrts/")
    vrt_dir.mkdir(exist_ok=True)
    mosaic_path = vrt_dir.joinpath("npi_mosaic.vrt")

    # Mosaic the tiles in a VRT
    gdal.BuildVRT(str(mosaic_path), uris)

    # Warp the VRT into one TIF
    gdal.Warp(
        str(output_path),
        str(mosaic_path),
        dstSRS=crs.to_wkt(),
        creationOptions=get_dem_co(),
        xRes=res[0],
        yRes=res[1],
        resampleAlg=rasterio.warp.Resampling.cubic_spline,
        outputBounds=list(bounds),
        multithread=True,
    )


def prepare_mask(filepath: Path) -> np.ndarray:
    """
    Load and modify the mask of an ArcticDEM to be more conservative.

    1. Dilate the excluded regions to exclude the periphery and remove lone pixels.
    2. Remove all disconnected patches with an area of less than 1% of the largest patch.

    Arguments
    ---------
    - filepath: The path to the mask to load and modify

    Returns
    -------
    A boolean outlier mask (True if a pixel should be excluded) array

    """
    outlier_mask = gu.Raster(str(filepath)).data.filled(0) == 0

    # Dilate the mask (expand the excluded areas)
    struct = scipy.ndimage.iterate_structure(scipy.ndimage.generate_binary_structure(2, 1), 3)
    outlier_mask = scipy.ndimage.binary_dilation(outlier_mask.squeeze(), struct)

    # Label each connected inlier component
    labelled, _ = scipy.ndimage.label(~outlier_mask)
    # Count the numbers of unique labels
    unique, counts = np.unique(labelled, return_counts=True)
    # Zero is assumed to be the label of the outliers (which we want to exclude from this part)
    counts = counts[unique != 0]
    unique = unique[unique != 0]
    # Create a new mask which is False (inliers) for all patches that are larger than 1% of the largest patch
    # All small patches and the previous outliers are True (outliers)
    outlier_mask = np.isin(labelled, unique[counts > int(counts[np.argmax(counts)] * 0.01)], invert=True)

    return outlier_mask


def prepare_arcticdem(dem_path: Path) -> tuple[Path, Path]:
    """
    Prepare warped and cropped VRTs for one ArcticDEM.

    Arguments
    ---------
    - dem_path: The filepath to the ArcticDEM

    Returns
    -------
    A tuple containing a warped VRT of the DEM and its mask (DEM, mask).
    """
    temp_dir = get_temp_dir()
    crs = get_crs()
    res = get_res()
    bounds = get_bounds(res=res)

    ad_vrt_dir = temp_dir.joinpath("arcticdem_vrts")
    ad_vrt_dir.mkdir(exist_ok=True)

    mask_path = dem_path.with_stem(dem_path.stem.replace("_dem", "_matchtag"))

    vrts = []

    for filepath in [dem_path, mask_path]:
        # First create a warped VRT to convert the CRS
        warp_vrt_path = ad_vrt_dir.joinpath(filepath.stem + f"_epsg{crs.to_epsg()}.vrt")
        create_warped_vrt(str(filepath), str(warp_vrt_path), out_crs=crs)

        # Then, build a new VRT to resample and crop the raster
        out_path = warp_vrt_path.with_stem(warp_vrt_path.stem + f"_{res[0]}m")
        gdal.BuildVRT(
            str(out_path),
            str(warp_vrt_path),
            outputBounds=list(bounds),
            xRes=res[0],
            yRes=res[1],
            resampleAlg="bilinear",
        )
        vrts.append(out_path)

        # gdal.Warp(
        #     str(out_path),
        #     str(filepath),
        #     xRes=5,
        #     yRes=5,
        #     dstSRS=crs.to_wkt(),
        #     outputBounds=list(bounds_5m()),
        #     resampleAlg=rasterio.warp.Resampling.min if filepath == mask_path else rasterio.warp.Resampling.bilinear,
        #     creationOptions=CREATION_OPTIONS,
        #     multithread=True,
        # )
    return vrts[0], vrts[1]


def now_time() -> str:
    """Return a string showing the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def coregister(dem_path: Path, verbose: bool = True):
    """
    Co-register an ArcticDEM to the NPI mosaic.

    The used approach is ICP for large translation/rotation and then NuthKaab for sub-pixel alignment.
    The stable terrain mask is used to exclude potential unstable terrain.
    Co-registration parameters are saved alongside the co-registered DEM.

    Arguments
    ---------
    - dem_path: The filepath to the ArcticDEM
    - verbose: Whether to print updates to the console

    Returns
    -------
    The path to the co-registered ArcticDEM
    """
    temp_dir = get_temp_dir()
    coreg_dir = temp_dir.joinpath("arcticdem_coreg")

    output_path = coreg_dir.joinpath(dem_path.stem + "_coreg.tif")
    if output_path.is_file():
        return output_path

    coreg_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"{now_time()}: Processing {dem_path.name}")

    dem_vrt_path, mask_vrt_path = prepare_arcticdem(dem_path=dem_path)

    mask = prepare_mask(mask_vrt_path)
    if verbose:
        print(f"{now_time()}: Loaded and modified mask")

    stable_terrain_path = build_stable_terrain_mask(verbose=verbose)
    stable_terrain_mask = gu.Raster(str(stable_terrain_path)).data.filled(0) == 1
    if verbose:
        print(f"{now_time()}: Loaded stable terrain mask")

    tba_dem = gu.Raster(str(dem_vrt_path))
    tba_dem.set_mask(mask)
    if verbose:
        print(f"{now_time()}: Loaded TBA DEM")

    del mask

    npi_mosaic_path = build_npi_mosaic(verbose=verbose)
    ref_dem = gu.Raster(str(npi_mosaic_path))

    if verbose:
        print(f"{now_time()}: Loaded ref. DEM. Running co-registration")

    coreg = xdem.coreg.ICP() + xdem.coreg.NuthKaab()
    coreg.fit(
        reference_dem=ref_dem.data,
        dem_to_be_aligned=tba_dem.data,
        transform=ref_dem.transform,
        inlier_mask=stable_terrain_mask,
    )
    del stable_terrain_mask
    del ref_dem
    if verbose:
        print(f"{now_time()}: Finished co-registration")

    out_meta = output_path.with_suffix(".json")
    metadata = []
    for part in coreg.pipeline:
        metadata.append(
            {
                "name": re.search(r"coreg.*'", str(part.__class__)).group().replace("'", "").replace("coreg.", ""),
                "meta": part.__dict__,
            }
        )
    with open(out_meta, "w") as outfile:
        json.dump(metadata, outfile, cls=NumpyArrayEncoder)

    tba_dem = coreg.apply(tba_dem)

    if verbose:
        print(f"{now_time()}: Applied co-registration")

    tba_dem.save(str(output_path), co_opts=get_xdem_dem_co())

    if verbose:
        print(f"{now_time()}: Saved {output_path.name}")


def generate_difference(dem_path: Path, verbose: bool = False):

    temp_dir = get_temp_dir()
    dh_dir = temp_dir.joinpath("dh")

    output_path = dh_dir.joinpath(dem_path.stem + "_dh.tif")
    if output_path.is_file():
        return output_path

    dh_dir.mkdir(exist_ok=True, parents=True)

    npi_mosaic_path = build_npi_mosaic(verbose=verbose)
    npi_dem = xdem.DEM(str(npi_mosaic_path))

    dem = xdem.DEM(str(dem_path), read_from_fn=False)
    (dem - npi_dem).save(output_path, co_opts=get_xdem_dem_co())

    return output_path


def main():

    dem_paths = Path("data/ArticDEM/").glob("*_dem.tif")

    for dem_path in dem_paths:
        dem_coreg = coregister(dem_path=dem_path)
        generate_difference(dem_coreg)

    # dem0 = Path("data/SETSM_s2s041_WV01_20210705_10200100B32D1400_10200100B37B2C00_2m_lsf_seg2_dem.tif")
    # dem1 = Path("data/ArticDEM/SETSM_s2s041_WV01_20210625_10200100B385AF00_10200100B3AF3500_2m_lsf_seg1_dem.tif")


if __name__ == "__main__":
    main()
