from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def create_dem_blend(key: str, bounds: dict[str, float | int], start_year: int, fill_value: float = 0.0):
    dirname = Path(f"temp/subsets/{key}/")

    out_path = dirname / f"{key}_dem_{start_year}.tif"

    if out_path.is_file():
        return out_path

    import rasterio.fill
    def get_dem_path(year) -> Path:
        return Path(f"temp.svalbard/medians/svalbard/dem/median_filt_075_dem_{year}.vrt")

    bounds = rasterio.coords.BoundingBox(**bounds)
    # start_year: int = data[key]["start_year"]
    # fill_value: float = data[key].get("fill_value", 0.0)

    with rasterio.open(get_dem_path(start_year)) as raster:
        window = rasterio.windows.from_bounds(*bounds, transform=raster.transform)

        arr = raster.read(1, window=window, masked=True).filled(np.nan)

        # print(window)

    n_missing = 0
    for year in reversed(range(2012, start_year)):
        missing_data = ~np.isfinite(arr)

        n_missing = np.count_nonzero(missing_data)
        if n_missing == 0:
            break

        with rasterio.open(get_dem_path(year)) as raster:
            arr[missing_data] = raster.read(1, window=window, masked=True).filled(
                np.nan
            )[missing_data]


    # plt.imshow(arr)
    # plt.show()
    # return
    #
    arr = rasterio.fill.fillnodata(arr, mask=np.isfinite(arr))

    if n_missing != 0:
        arr[~np.isfinite(arr)] = fill_value

    with rasterio.open(get_dem_path(start_year)) as raster0:
        transform = rasterio.windows.transform(window, raster0.transform)
        dirname.mkdir(exist_ok=True, parents=True)
        with rasterio.open(
            out_path,
            "w",
            **(
                raster.meta
                | {
                    "driver": "GTiff",
                    "transform": transform,
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                }
            ),
            compress="deflate",
            zlevel=12,
            tiled=True,
        ) as raster:
            raster.write(arr, 1)
    return out_path
    

def main(key: str = "midtre"):
    data = {
        "scheele": {
            "bounds": {
                "left": 541343,
                "top": 8636570,
                "right": 552827,
                "bottom": 8618337,
            },
            "start_year": 2023,
        },
        "midtre": {
            "bounds": {
                "left": 433581,
                "top": 8762024,
                "right": 438553,
                "bottom": 8755819,
            },
            "start_year": 2015,
        },
        "ganskij": {
            "bounds": {
                "left": 619351,
                "bottom": 8741098,
                "right": 629478,
                "top": 8751034,
            },
            "start_year": 2021,
        },
        "vallakra": {
            "bounds": {
                "left": 544202,
                "bottom": 8637040,
                "right": 556607,
                "top": 8650491,
            },
            "start_year": [2023, 2022, 2021],
        },
        "kongsvegen": {
            "bounds": {
                "left": 441044,
                "bottom": 8738390,
                "right": 470973,
                "top": 8759470,
            },
            "start_year": 2021,
        },
        "finsterwalderbreen_antoniabreen": {
            "bounds": {
                "left": 496500,
                "bottom": 8592977,
                "right": 510000,
                "top": 8607820,
            },
            "start_year": 2023,
        },
        "filantropbreen": {
            "bounds": {
                "left": 537680,
                "bottom": 8624689,
                "right": 543200,
                "top": 8628106
            },
            "start_year": 2023,
        },
        "edvard_mette_ragna_kropp": {
            "bounds": {
                "left": 550800,
                "bottom": 8632900,
                "right": 567570,
                "top": 8654150,
            },
            "start_year": 2024,
        },
        "slakbreen": {
            "bounds": {
                "left": 527350,
                "bottom": 8648700,
                "right": 541840,
                "top": 8659800,
            },
            "start_year": [2024, 2023, 2022],
        },
        "fimbulisen": {
            "bounds": {
                "left": 554200,
                "bottom": 8695730,
                "right": 566300,
                "top": 8705000,
            },
            "start_year": 2022,
        },
        "dron_moysal_kok": {
            "bounds": {
                "left": 528900,
                "bottom": 8663600,
                "right": 545780,
                "top": 8677830,
            },
            "start_year": 2022,
        },
        "dronbreen": {
            "bounds": {
                "left": 536760,
                "bottom": 8667680,
                "right": 544460,
                "top": 8677650,
            },
            "start_year": 2024,
        },
        "jinnbreen": {
            "bounds": {
                "left": 554600,
                "bottom": 8671430,
                "right": 562860,
                "top": 8680190,
            },
            "start_year": [2022, 2024],
        },
        "elfenbeinbreen": {
            "bounds": {
                "left": 567890,
                "bottom": 8674000,
                "right": 577200,
                "top": 8686380,
            },
            "start_year": 2022,
        },
        "rabotbreen": {
            "bounds": {
                "left": 564340,
                "bottom": 8689100,
                "right": 576000,
                "top": 8705500,
            },
            "start_year": 2024,
        },
        "von_postbreen": {
            "bounds": {
                "left": 552630,
                "bottom": 8699800,
                "right": 572100,
                "top": 8720900,
            },
            "start_year": 2024,
        },
        "austfonna": {
            "bounds": {
                "left": 631850,
                "bottom": 8851500,
                "right": 656800,
                "top": 8885900,
            },
            "start_year": 2024,
        },
        "scott_turnerbreen": {
            "bounds": {
                "left": 519440,
                "bottom": 8667170,
                "right": 523520,
                "top": 8672170,
            },
            "start_year": 2021,
        },
        "aldegondabreen": {
            "bounds": {
                "left": 475830,
                "bottom": 8653480,
                "right": 483020,
                "top": 8657940,
            },
            "start_year": 2021,
        },
        "gronfjordbreane": {
            "bounds": {
                "left": 478020,
                "bottom": 8644440,
                "right": 487570,
                "top": 8653320,
            },
            "start_year": [2021, 2013],
        },
        "renardbreen": {
            "bounds": {
                "left": 479650,
                "bottom": 8597900,
                "right": 490080,
                "top": 8607600,
            },
            "start_year": 2013,
        },
        "tillbergfonna": {
            "bounds": {
                "left": 512970,
                "bottom": 8662440,
                "right": 519960,
                "top": 8668260,
            },
            "start_year": [2024, 2023, 2022, 2021],
        },
    }

    for key, params in data.items():
        start_years = params["start_year"]
        if isinstance(start_years, int):
            start_years = [start_years]

        for start_year in start_years:
            print(key, start_year)
            create_dem_blend(key=key, **(params | {"start_year": start_year}))

    # bounds = rasterio.coords.BoundingBox(left=541343, top=8636570, right=552827, bottom=8618337)
    #


if __name__ == "__main__":
    main()
