"""Repair of the day-of-year band in the annual mosaics.

Until 2026-10-09, ``stacking.py`` encoded each strip's acquisition date as
``month * 30 + day`` rather than as the day of year, and ``mosaicking.py`` initialised
the mosaic band with ones, so pixels without any DEM held 1 instead of the nodata value
0. Reprocessing the stacks to fix this would take days, so the exported
``dem_<year>_doy.tif`` rasters are instead converted here.

The legacy encoding is inverted month by month. It is exact for single acquisitions
except that day 31 of a month collides with day 1 of the next (e.g. 31 January and
1 February both encode as 61) and decodes as the latter, one day late. Per-pixel medians
of an even number of strips are means of two encoded dates, which decode to within
about a day of the true mean date.
"""

import calendar
import os
import uuid
from pathlib import Path
from typing import Final

import numpy as np
import rasterio
import rasterio.windows
import tqdm

#: GDAL metadata tag written on converted rasters, so a second conversion is refused.
ENCODING_TAG: Final = "DOY_ENCODING"
ENCODING_VALUE: Final = "day_of_year"
#: Exported mosaics directory, as written by ``export_mosaics.sh``.
MOSAIC_DIR: Final = Path("temp.svalbard/filt/svalbard/mosaics_3584/out")
YEARS: Final = tuple(range(2013, 2025))


def legacy_to_doy(encoded: np.ndarray, year: int) -> np.ndarray:
    """Convert legacy ``month * 30 + day`` values to the day of year.

    Parameters
    ----------
    encoded
        Legacy-encoded dates. Values below 31 (no valid date encodes below 31) are
        treated as nodata and returned as 0.
    year
        Calendar year, for leap-year handling.

    Returns
    -------
    Day of year (1 = 1 January) as uint16, 0 where nodata.

    Examples
    --------
    >>> legacy_to_doy(np.array([0, 1, 31, 61, 7 * 30 + 15, 12 * 30 + 31]), 2015).tolist()
    [0, 0, 1, 32, 196, 365]
    >>> legacy_to_doy(np.array([3 * 30 + 1]), 2016).tolist()
    [61]
    """
    days_in_month = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    days_before_month = np.concatenate([[0], np.cumsum(days_in_month)[:-1]])

    values = encoded.astype("int64")
    month = np.clip((values - 1) // 30, 1, 12)
    day = values - 30 * month
    doy = days_before_month[month - 1] + day

    return np.where(values >= 31, doy, 0).astype("uint16")


def convert_legacy_doy_raster(src_path: Path, count_path: Path, dst_path: Path) -> None:
    """Write a day-of-year raster converted from a legacy-encoded one.

    Pixels where the matching ``count`` raster is 0 are set to the nodata value 0. The
    output is written to a temporary sibling and atomically moved to ``dst_path``.
    Overviews and statistics are left to the caller (``gdaladdo``, ``gdal_edit -stats``),
    as in ``export_mosaics.sh``.

    Raises
    ------
    ValueError
        If ``src_path`` is already tagged as converted.
    """
    year = int(src_path.stem.split("_")[1])
    tmp_path = dst_path.with_name(f".{dst_path.stem}.{uuid.uuid4().hex}.tmp.tif")

    try:
        with rasterio.open(src_path) as src, rasterio.open(count_path) as count_raster:
            if src.tags().get(ENCODING_TAG) == ENCODING_VALUE:
                raise ValueError(f"{src_path} is already converted to day of year")
            if (src.width, src.height, src.transform) != (count_raster.width, count_raster.height, count_raster.transform):
                raise ValueError(f"{src_path} and {count_path} are not on the same grid")

            profile = src.profile | {
                "nodata": 0,
                "compress": "DEFLATE",
                "zlevel": 9,
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "bigtiff": "YES",
                "num_threads": "ALL_CPUS",
            }
            with rasterio.open(tmp_path, "w", **profile) as dst:
                windows = [window for _, window in src.block_windows(1)]
                for window in tqdm.tqdm(windows, desc=src_path.name, smoothing=0.05):
                    doy = legacy_to_doy(src.read(1, window=window), year=year)
                    doy[count_raster.read(1, window=window) == 0] = 0
                    dst.write(doy, 1, window=window)
                dst.update_tags(**{ENCODING_TAG: ENCODING_VALUE})
                dst.update_tags(1, units="day of year", description="Median acquisition day of year (1 = 1 January)")
                dst.set_band_description(1, "Median acquisition day of year (1 = 1 January)")

        os.replace(tmp_path, dst_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def convert_mosaic_dir(mosaic_dir: Path = MOSAIC_DIR) -> None:
    """Convert every exported ``dem_<year>_doy.tif`` in place.

    The legacy rasters are kept as ``dem_<year>_doy_legacy.tif``. Rebuild overviews and
    statistics afterwards, e.g.
    ``gdaladdo -r average dem_<year>_doy.tif && gdal_edit -stats dem_<year>_doy.tif``.
    """
    for year in YEARS:
        path = mosaic_dir / f"dem_{year}_doy.tif"
        legacy_path = path.with_stem(path.stem + "_legacy")
        if not legacy_path.is_file():
            path.replace(legacy_path)
        elif path.is_file():
            with rasterio.open(path) as raster:
                if raster.tags().get(ENCODING_TAG) == ENCODING_VALUE:
                    continue
        convert_legacy_doy_raster(legacy_path, mosaic_dir / f"dem_{year}_count.tif", path)


if __name__ == "__main__":
    convert_mosaic_dir()
