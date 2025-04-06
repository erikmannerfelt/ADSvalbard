from pathlib import Path

import projectfiles


class Constants(projectfiles.Constant):
    cache_dir = Path(__file__).parent.parent.joinpath(".cache")
    data_dir = Path(__file__).parent.parent.joinpath("data")
    temp_dir = Path(__file__).parent.parent.joinpath("temp")
    res: float = 5.0
    coreg_res: float = 5.0
    crs_epsg: int = 32633

    subregions: dict[list[float]] = {
        "dronbreen": [536837.5, 8667472.5, 544677.5, 8679002.5],
        "tinkarp": [548762.5, 8655937.5, 553272.5, 8659162.5],
    }
    regions: dict[str, dict[str, float]] = {
        "svalbard": {
            "left": 341002.5,
            "bottom": 8455002.5,
            "right": 905002.5,
            "top": 8982002.5,
        },
        "nordenskiold": {
            "left": 443002.5,
            "bottom": 8626007.5,
            "right": 560242.5,
            "top": 8703007.5,
        },
        "heerland": {
            "left": 537010,
            "bottom": 8602780,
            "right": 582940,
            "top": 8659000,
        },
        "recherchefront": {
            "left": 489290,
            "bottom": 8595240,
            "right": 499670,
            "top": 8604340,
        },
    }


CONSTANTS = Constants()
