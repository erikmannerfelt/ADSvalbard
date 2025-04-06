import os
from pathlib import Path

from tqdm import tqdm

import adsvalbard.utilities
from adsvalbard.constants import CONSTANTS


def get_data_urls():
    np_dem_base_url = "https://public.data.npolar.no/kartdata/S0_Terrengmodell/Delmodell/"

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
                "NP_S0_DTM5_2008_13657_35.zip",
                "NP_S0_DTM5_2008_13659_33.zip",
                "NP_S0_DTM5_2008_13660_33.zip",
                "NP_S0_DTM5_2008_13666_35.zip",
                "NP_S0_DTM5_2008_13667_35.zip",
                "NP_S0_DTM5_2009_13822_33.zip",
                "NP_S0_DTM5_2009_13824_33.zip",
                "NP_S0_DTM5_2009_13827_35.zip",
                "NP_S0_DTM5_2009_13833_33.zip",
                "NP_S0_DTM5_2009_13835_33.zip",
                "NP_S0_DTM5_2010_13826_33.zip",
                "NP_S0_DTM5_2010_13828_33.zip",
                "NP_S0_DTM5_2010_13832_35.zip",
                "NP_S0_DTM5_2010_13836_33.zip",
                "NP_S0_DTM5_2010_13918_33.zip",
                "NP_S0_DTM5_2010_13920_35.zip",
                "NP_S0_DTM5_2010_13922_33.zip",
                "NP_S0_DTM5_2010_13923_33.zip",
                "NP_S0_DTM5_2011_13831_35.zip",
                "NP_S0_DTM5_2011_25160_33.zip",
                "NP_S0_DTM5_2011_25161_33.zip",
                "NP_S0_DTM5_2011_25162_33.zip",
                "NP_S0_DTM5_2011_25163_33.zip",
                "NP_S0_DTM5_2012_25235_33.zip",
                "NP_S0_DTM5_2012_25236_35.zip",
                "NP_S0_DTM5_2021_33.zip",
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

            urls[key][i] = (url, CONSTANTS.data_dir.joinpath(key).joinpath(filename))

    return urls


def download_category(key: str) -> list[Path]:

    urls = get_data_urls()[key]

    finished = []
    to_download = []
    for url, filepath in urls:
        if filepath.is_file():
            finished.append(filepath)
        else:
            to_download.append((url, filepath))

    if len(to_download) > 0:
        for url, filepath in tqdm(to_download, desc=f"Downloading {key}"):

            filepath.parent.mkdir(exist_ok=True, parents=True)
            adsvalbard.utilities.download_large_file(url, filepath.name, filepath.parent)
            finished.append(filepath)

    finished.sort()

    return finished
